import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

from src.config import N_CLUSTERS
from src.clustering.fuzzy_clusterer import FuzzyClusterer
from src.data.preprocessor import preprocess_corpus
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_store import ChromaStore


def main() -> None:
    print("=" * 60)
    print("  Trademarkia AI — Corpus Setup (Parts 1 & 2)")
    print("=" * 60)

    print("\n[1/5] Loading 20 Newsgroups dataset …")
    raw = fetch_20newsgroups(
        subset="all",
        remove=(),
        shuffle=True,
        random_state=42,
    )
    raw_texts: list[str] = raw.data
    categories: list[str] = [raw.target_names[t] for t in raw.target]
    print(f"   Loaded {len(raw_texts)} raw documents.")

    print("\n[2/5] Cleaning corpus …")
    texts, cats = preprocess_corpus(raw_texts, categories)
    print(f"   {len(texts)} documents retained after cleaning.")

    print("\n[3/5] Embedding documents …")
    embedder = Embedder()
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress=True,
    )
    print(f"   Embeddings shape: {embeddings.shape}")

    print("\n[4/5] Fitting fuzzy clustering (GMM with BIC sweep) …")
    clusterer = FuzzyClusterer()
    clusterer.fit(embeddings, sweep=True, k_range=range(10, 31))
    print(f"   Selected K = {clusterer.n_clusters} clusters.")

    print("   Computing soft cluster assignments for all documents …")
    all_probs = clusterer.get_all_probabilities(embeddings)
    dominant_clusters = np.argmax(all_probs, axis=1)

    print("\n[5/5] Storing documents + embeddings in ChromaDB …")
    store = ChromaStore()

    doc_ids = [f"doc_{i:06d}" for i in range(len(texts))]
    metadatas = [
        {
            "category": cats[i],
            "dominant_cluster": int(dominant_clusters[i]),
            "cluster_top1_prob": float(np.sort(all_probs[i])[-1]),
            "cluster_top2_prob": float(np.sort(all_probs[i])[-2]),
            "cluster_entropy": float(
                -np.sum(all_probs[i] * np.log(all_probs[i] + 1e-10))
            ),
        }
        for i in range(len(texts))
    ]

    store.add_documents(
        doc_ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print("\n" + "=" * 60)
    print(f"  Setup complete!")
    print(f"  Corpus size  : {store.count()} documents")
    print(f"  Clusters (K) : {clusterer.n_clusters}")
    print(f"  Embedding dim: {embedder.dim}")
    print("  ChromaDB     : ./data/chroma_db/")
    print("  GMM model    : ./data/gmm_model.joblib")
    print("  PCA model    : ./data/pca_model.joblib")
    print("=" * 60)
    print("\nRun the API with:")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
