"""Stdlib HTTP retrieval server for the web experiment."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re

import numpy as np
import pandas as pd


def to_jsonable(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, set):
        return [to_jsonable(x) for x in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def _normalize_query_key(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class RetrieverService:
    def __init__(self, parquet_path: str, cache_file: str, question_col: str = "question"):
        self.parquet_path = parquet_path
        self.cache_file = cache_file
        self.question_col = question_col
        self.df = pd.read_parquet(self.parquet_path).reset_index(drop=True)

        if self.question_col not in self.df.columns:
            raise ValueError(
                f"Column '{self.question_col}' not found in parquet: {self.parquet_path}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        cache = np.load(self.cache_file, allow_pickle=True)
        cache_questions = cache["questions"].tolist()
        cache_embeddings = cache["embeddings"].astype(np.float32)

        self.query2embedding = {}
        for q, emb in zip(cache_questions, cache_embeddings):
            q = _normalize_query_key(q)
            if q not in self.query2embedding:
                self.query2embedding[q] = emb

        db_embeddings = []
        db_row_indices = []
        missing_questions = []
        parquet_questions = self.df[self.question_col].astype(str).tolist()

        for row_idx, q in enumerate(parquet_questions):
            emb = self.query2embedding.get(_normalize_query_key(q))
            if emb is None:
                missing_questions.append((row_idx, q))
                continue
            db_embeddings.append(emb)
            db_row_indices.append(row_idx)

        if not db_embeddings:
            raise ValueError(
                f"No rows in parquet matched cache by question column '{self.question_col}'. "
                f"Parquet: {self.parquet_path}"
            )
        if missing_questions:
            raise ValueError(
                f"{len(missing_questions)} questions in parquet were not found in cache. "
                f"First few missing entries: {missing_questions[:10]}"
            )

        self.db_embeddings = np.stack(db_embeddings).astype(np.float32)
        self.db_row_indices = np.array(db_row_indices, dtype=np.int64)

        norms = np.linalg.norm(self.db_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        self.db_embeddings = self.db_embeddings / norms

    def get_query_embedding(self, query: str) -> np.ndarray:
        query = _normalize_query_key(query)
        if query not in self.query2embedding:
            raise KeyError(f"Query not found in cache: {query}")
        emb = self.query2embedding[query]
        norm = max(float(np.linalg.norm(emb)), 1e-12)
        return (emb / norm).astype(np.float32)

    def _rows_for_indices(self, local_indices) -> list[dict]:
        rows = []
        for local_idx in local_indices:
            row_idx = int(self.db_row_indices[int(local_idx)])
            row = to_jsonable(self.df.iloc[row_idx].to_dict())
            row["_index"] = row_idx
            rows.append(row)
        return rows

    def search(self, query: str, topk: int = 5) -> list[dict]:
        q_emb = self.get_query_embedding(query)
        scores = self.db_embeddings @ q_emb
        topk = min(int(topk), len(scores))
        topk_idx = np.argsort(-scores)[:topk]

        results = []
        for local_idx, row in zip(topk_idx, self._rows_for_indices(topk_idx)):
            row["_score"] = float(scores[int(local_idx)])
            results.append(row)
        return results

    def search_least_relevant(self, query: str, topk: int = 1) -> list[dict]:
        q_emb = self.get_query_embedding(query)
        scores = self.db_embeddings @ q_emb
        topk = min(int(topk), len(scores))
        bottomk_idx = np.argsort(scores)[:topk]

        results = []
        for local_idx, row in zip(bottomk_idx, self._rows_for_indices(bottomk_idx)):
            row["_score"] = float(scores[int(local_idx)])
            results.append(row)
        return results


SERVICE: RetrieverService | None = None


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {"status": "ok"})
            return
        _json_response(self, 404, {"ok": False, "error": f"Unknown route: GET {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            _json_response(self, 400, {"ok": False, "error": "Invalid JSON payload"})
            return

        if not isinstance(payload, dict):
            _json_response(self, 400, {"ok": False, "error": "Payload must be a JSON object"})
            return

        if self.path not in {"/search", "/search_least_relevant"}:
            _json_response(self, 404, {"ok": False, "error": f"Unknown route: POST {self.path}"})
            return

        query = str(payload.get("query") or "")
        topk = int(payload.get("topk") or 5)
        if not query:
            _json_response(self, 400, {"ok": False, "error": "Missing query"})
            return

        assert SERVICE is not None
        try:
            if self.path == "/search":
                results = SERVICE.search(query, topk)
            else:
                results = SERVICE.search_least_relevant(query, topk)
        except KeyError as exc:
            _json_response(
                self,
                200,
                {
                    "query": query,
                    "topk": topk,
                    "results": [],
                    "warning": str(exc),
                },
            )
            return
        except ValueError as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
            return

        _json_response(
            self,
            200,
            {
                "query": query,
                "topk": topk,
                "results": results,
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    global SERVICE

    parser = argparse.ArgumentParser(description="Run the web retrieval server.")
    parser.add_argument("--parquet", required=True, help="Database parquet file")
    parser.add_argument("--cache_file", required=True, help="Unified embedding cache file")
    parser.add_argument("--port", type=int, default=9011)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--question_col", default="question")
    args = parser.parse_args()

    SERVICE = RetrieverService(
        parquet_path=args.parquet,
        cache_file=args.cache_file,
        question_col=args.question_col,
    )
    server = ThreadingHTTPServer((args.host, args.port), _RequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
