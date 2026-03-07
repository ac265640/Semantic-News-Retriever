from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL


class Embedder:
    _instance: "Embedder | None" = None

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print(f"[Embedder] Loading model '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready — embedding dim={self.dim}")

    @classmethod
    def get_instance(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
