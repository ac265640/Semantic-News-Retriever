import chromadb
from chromadb.config import Settings
from typing import Optional
import numpy as np

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        #Use HNSW index
#Use cosine similarity as distance metric
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[ChromaStore] Collection '{collection_name}' has "
            f"{self.collection.count()} documents."
        )

    def add_documents(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 512,
    ) -> None:
        n = len(doc_ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            self.collection.upsert(
                ids=doc_ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        print(f"[ChromaStore] Upserted {n} documents.")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray, list[dict]]:
        result = self.collection.get(include=["embeddings", "metadatas"])
        ids = result["ids"]
        embeddings = np.array(result["embeddings"], dtype=np.float32)
        metadatas = result["metadatas"]
        return ids, embeddings, metadatas

    def count(self) -> int:
        return self.collection.count()

    def update_metadata_batch(
        self, doc_ids: list[str], metadatas: list[dict]
    ) -> None:
        self.collection.update(ids=doc_ids, metadatas=metadatas)
