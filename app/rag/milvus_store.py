from __future__ import annotations
from typing import List, Dict, Any

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from app.config import MILVUS_URI

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class MilvusStore:
    """
    Minimal store for:
    - create collection (if missing)
    - insert rows (id, vector, text, metadata fields)
    """
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = MilvusClient(MILVUS_URI)
        self.embedder = SentenceTransformer(_EMBED_MODEL)

    def ensure_collection(self, dim: int) -> None:
        if self.client.has_collection(self.collection_name):
            return

        # Create a collection with a vector field
        # (MilvusClient simplifies setup; we’ll add indexes later.)
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim,
            metric_type="IP",  # use inner product since we normalize embeddings
            consistency_level="Strong",
        )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        vectors = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.ensure_collection(dim=len(vectors[0]))

        rows = []
        for _id, vec, meta, text in zip(ids, vectors, metadatas, texts):
            rows.append({
                "id": _id,
                "vector": vec,
                "text": text,
                "path": meta.get("path"),
                "chunk_id": int(meta.get("chunk_id", 0)),
                "ext": meta.get("ext"),
            })

        self.client.insert(collection_name=self.collection_name, data=rows)
        return len(rows)
