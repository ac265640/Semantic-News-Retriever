import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(ROOT_DIR / "data" / "chroma_db")
)
CHROMA_COLLECTION_NAME: str = "newsgroups_corpus"

CLUSTER_MODEL_PATH: str = os.getenv(
    "CLUSTER_MODEL_PATH", str(ROOT_DIR / "data" / "gmm_model.joblib")
)
PCA_MODEL_PATH: str = os.getenv(
    "PCA_MODEL_PATH", str(ROOT_DIR / "data" / "pca_model.joblib")
)
N_CLUSTERS: int = int(os.getenv("N_CLUSTERS", "20"))
PCA_N_COMPONENTS: int = 50

CACHE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.75")
)

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
