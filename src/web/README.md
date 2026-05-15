# Web Experiment

Mind2Web-backed web experiment package.

## What It Contains

- Offline replay web environment server based on Mind2Web tasks (in top-level `env/web/`)
- OpenClaw-style compact page observations with `ref=eN` action handles
- Mind2Web-aligned action space: `CLICK`, `TYPE`, `SELECT`
- Plain GRPO baseline under `src.web.baseline`
- Rollout pipeline, evaluation helpers, retrieval wrappers

## Expected Data Layout

Data lives under `data/web/`:

```text
data/web/
├── Mind2Web_dataset/
├── train.parquet                  # columns: id, question, prompt, metadata
├── train.jsonl                    # same schema as train.parquet
├── test_task.parquet              # official cross-task split
├── test_task.jsonl
├── test_website.parquet           # official cross-website split
├── test_website.jsonl
├── test_domain.parquet            # official cross-domain split
├── test_domain.jsonl
├── all.parquet                    # train + all official test splits
├── all.jsonl
└── cache/
    └── unified_embeddings.npz
```

`prompt` is a chat-message array like ALFWorld's parquet format, so slime can
consume it with `--input-key prompt --label-key metadata --apply-chat-template`.

Training retrieval should use `train.parquet`, while test-time retrieval should
use `all.parquet`.

See scripts in `scripts/web/`.
