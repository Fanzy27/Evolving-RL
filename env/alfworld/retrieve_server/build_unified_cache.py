import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]


class QwenEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"Loading model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquets", nargs="+", required=True, help="List of parquet files")
    parser.add_argument("--output", required=True, help="Unified cache output file, e.g. cache/unified_embeddings.npz")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    all_questions = []
    all_ids = []
    all_row_indices = []
    all_parquet_paths = []

    for parquet_path in args.parquets:
        print(f"Reading {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if args.question_col not in df.columns:
            raise ValueError(
                f"Column '{args.question_col}' not found in {parquet_path}. "
                f"Available columns: {df.columns.tolist()}"
            )

        questions = df[args.question_col].astype(str).tolist()
        ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
        row_indices = list(range(len(df)))
        parquet_paths = [os.path.abspath(parquet_path)] * len(df)

        all_questions.extend(questions)
        all_ids.extend(ids)
        all_row_indices.extend(row_indices)
        all_parquet_paths.extend(parquet_paths)

    print(f"Total samples: {len(all_questions)}")

    embedder = QwenEmbedder(model_name=args.model_name)
    all_embeddings = embedder.encode(all_questions, batch_size=args.batch_size).astype(np.float32)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        questions=np.array(all_questions, dtype=object),
        ids=np.array(all_ids, dtype=object),
        row_indices=np.array(all_row_indices, dtype=np.int64),
        parquet_paths=np.array(all_parquet_paths, dtype=object),
        embeddings=all_embeddings,
    )

    print(f"Unified cache saved to: {args.output}")


if __name__ == "__main__":
    main()
