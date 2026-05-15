"""ALFWorld task retrieval via the dense retrieve server.

Replaces the local registry approach with HTTP calls to retrieve_server,
which returns complete parquet rows — no separate lookup table needed.

Two retrieval modes:

  retrieve_tasks(args, query, K, exclude_path=None)
      Calls /search → top-K most semantically similar tasks.
      Used by the training rollout to find downstream evaluation tasks.

  retrieve_least_relevant_task(args, query, exclude_path=None)
      Calls /search_least_relevant → the task with the lowest cosine
      similarity to the query.
      Used by the eval ablation to test skill transfer across task types.

Both are async and return task_info dicts compatible with _task_info_to_sample
in rollout.py (full parquet rows with label/metadata fields intact).

Args URL is passed in explicitly by the caller.
"""

import os

from slime.utils.http_utils import post


def _row_to_path(row: dict) -> str | None:
    """Extract the registry-format directory path from a server result row.

    Tries game_file_path (direct column), then label.split/rel_dir, then
    metadata.split/rel_dir.  Returns None if no path can be constructed.
    """
    gfp = row.get("game_file_path")
    if gfp:
        return str(gfp)

    alfworld_data = "/mnt/tidal-alsh-share2/usr/fanzhiyuan2/projects/extractor/data/alfworld/official"

    label = row.get("label")
    if isinstance(label, dict):
        rel_dir = label.get("rel_dir", "")
        split = label.get("split", "train")
        if rel_dir:
            return os.path.join(alfworld_data, "json_2.1.1", split, rel_dir)

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        rel_dir = metadata.get("rel_dir", "")
        split = metadata.get("split", "train")
        if rel_dir:
            return os.path.join(alfworld_data, "json_2.1.1", split, rel_dir)

    return None


def _filter_exclude(results: list[dict], exclude_path: str | None) -> list[dict]:
    if not exclude_path:
        return results
    return [r for r in results if _row_to_path(r) != exclude_path]


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------


async def retrieve_tasks(
    query: str,
    K: int,
    url,
    exclude_path: str | None = None,
) -> list[dict]:
    """Return up to K most similar tasks from the retrieve server.

    Requests K + 5 results to have a buffer for client-side exclusion of the
    source task, then returns the first K after filtering.

    Args:
        query:        Task description string used as the semantic query.
        K:            Number of tasks to return.
        exclude_path: Registry-format directory path of the source task to
                      exclude (client-side filtering).

    Returns:
        List of task_info dicts (full parquet rows + _score, _index).
    """
    topk = K + 5  # buffer for exclusion filtering

    try:
        resp = await post(f"{url}/search", {"query": query, "topk": topk})
        results = resp.get("results", [])
    except Exception as exc:
        print(f"[retrieval] /search error for query '{query[:60]}': {exc}")
        return []

    results = _filter_exclude(results, exclude_path)
    return results[:K]


async def retrieve_least_relevant_task(
    query: str,
    url,
    exclude_path: str | None = None,
) -> dict | None:
    """Return the task with the lowest cosine similarity to the query.

    Args:
        query:        Task description string used as the semantic query.
        exclude_path: Registry-format directory path of the source task to
                      exclude (unlikely to be needed for least-relevant, but
                      included for correctness).

    Returns:
        A single task_info dict, or None if retrieval fails.
    """
    topk = 1 + (5 if exclude_path else 0)

    try:
        resp = await post(f"{url}/search_least_relevant", {"query": query, "topk": topk})
        results = resp.get("results", [])
    except Exception as exc:
        print(f"[retrieval] /search_least_relevant error for query '{query[:60]}': {exc}")
        return None

    results = _filter_exclude(results, exclude_path)
    return results[0] if results else None
