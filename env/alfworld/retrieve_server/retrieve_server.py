import os
import argparse
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


def to_jsonable(obj):
    import math
    import numpy as np
    import pandas as pd

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



class SearchRequest(BaseModel):
    query: str
    topk: int = 5


class RetrieverService:
    def __init__(self, parquet_path, cache_file, question_col="question"):
        self.parquet_path = os.path.abspath(parquet_path)
        self.cache_file = cache_file
        self.question_col = question_col

        self.df = pd.read_parquet(self.parquet_path).reset_index(drop=True)

        if self.question_col not in self.df.columns:
            raise ValueError(
                f"Column '{self.question_col}' not found in parquet: {self.parquet_path}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        cache = np.load(cache_file, allow_pickle=True)
        cache_questions = cache["questions"].tolist()
        cache_embeddings = cache["embeddings"].astype(np.float32)

        # 全局 question -> embedding
        self.query2embedding = {}
        for q, emb in zip(cache_questions, cache_embeddings):
            q = str(q)
            if q not in self.query2embedding:
                self.query2embedding[q] = emb

        # 根据当前 parquet 的 question 字段构建数据库
        db_embeddings = []
        db_row_indices = []
        db_questions = []
        missing_questions = []

        parquet_questions = self.df[self.question_col].astype(str).tolist()

        for row_idx, q in enumerate(parquet_questions):
            emb = self.query2embedding.get(q)
            if emb is None:
                missing_questions.append((row_idx, q))
                continue

            db_embeddings.append(emb)
            db_row_indices.append(row_idx)
            db_questions.append(q)

        if len(db_embeddings) == 0:
            raise ValueError(
                f"No rows in parquet matched cache by question column '{self.question_col}'. "
                f"Parquet: {self.parquet_path}"
            )

        if missing_questions:
            preview = missing_questions[:10]
            raise ValueError(
                f"{len(missing_questions)} questions in parquet were not found in cache. "
                f"First few missing entries: {preview}"
            )

        self.db_embeddings = np.stack(db_embeddings).astype(np.float32)
        self.db_row_indices = np.array(db_row_indices, dtype=np.int64)
        self.db_questions = np.array(db_questions, dtype=object)

        # 归一化
        norms = np.linalg.norm(self.db_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        self.db_embeddings = self.db_embeddings / norms

        print(f"Loaded parquet: {self.parquet_path}")
        print(f"Database size: {len(self.db_row_indices)}")
        print(f"Global cached queries: {len(self.query2embedding)}")

    def get_query_embedding(self, query: str):
        query = str(query)
        if query not in self.query2embedding:
            raise KeyError(f"Query not found in cache: {query}")
        emb = self.query2embedding[query]
        norm = np.linalg.norm(emb)
        norm = max(norm, 1e-12)
        emb = emb / norm
        return emb.astype(np.float32)

    def search(self, query: str, topk: int = 5):
        q_emb = self.get_query_embedding(query)
        scores = self.db_embeddings @ q_emb
        topk = min(topk, len(scores))
        topk_idx = np.argsort(-scores)[:topk]

        results = []
        for local_idx in topk_idx:
            row_idx = int(self.db_row_indices[local_idx])
            row = to_jsonable(self.df.iloc[row_idx].to_dict())
            row["_score"] = float(scores[local_idx])
            row["_index"] = int(row_idx)
            results.append(row)

        return results

    def search_least_relevant(self, query: str, topk: int = 1):
        q_emb = self.get_query_embedding(query)
        scores = self.db_embeddings @ q_emb
        topk = min(topk, len(scores))
        bottomk_idx = np.argsort(scores)[:topk]

        results = []
        for local_idx in bottomk_idx:
            row_idx = int(self.db_row_indices[local_idx])
            row = to_jsonable(self.df.iloc[row_idx].to_dict())
            row["_score"] = float(scores[local_idx])
            row["_index"] = int(row_idx)
            results.append(row)

        return results



def create_app(service: RetrieverService):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/search")
    def search(req: SearchRequest):
        try:
            results = service.search(req.query, req.topk)
            return {
                "query": req.query,
                "topk": req.topk,
                "results": results
            }
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/search_least_relevant")
    def search_least_relevant(req: SearchRequest):
        try:
            results = service.search_least_relevant(req.query, req.topk)
            return {
                "query": req.query,
                "topk": req.topk,
                "results": results
            }
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Database parquet file")
    parser.add_argument("--cache_file", required=True, help="Unified embedding cache file")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--question_col", default="question")
    args = parser.parse_args()

    service = RetrieverService(
        parquet_path=args.parquet,
        cache_file=args.cache_file,
        question_col=args.question_col
    )

    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
