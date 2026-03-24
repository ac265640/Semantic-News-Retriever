# Semantic-News-Retriever — Semantic Search System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-3.4-blue)](https://www.sbert.net)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple)](https://www.trychroma.com)
[![Python](https://img.shields.io/badge/Python-3.11-yellow)](https://python.org)

A lightweight semantic search system over the **20 Newsgroups dataset** (~18k news posts), featuring fuzzy clustering, a hand-built semantic cache, and a production-ready FastAPI service.

---

## Architecture Overview

```
20 Newsgroups Corpus
        │
        ▼
┌─────────────────┐     ┌──────────────────────┐
│  Preprocessor   │────▶│  all-MiniLM-L6-v2    │
│  (noise removal)│     │  (384-dim embeddings) │
└─────────────────┘     └──────────┬───────────┘
                                   │
              ┌────────────────────┼───────────────────┐
              ▼                    ▼                   ▼
    ┌─────────────────┐  ┌──────────────────┐ ┌──────────────────┐
    │    ChromaDB     │  │   GMM Fuzzy      │ │  Semantic Cache  │
    │  (vector store) │  │   Clusterer      │ │  (cluster-aware) │
    │  + HNSW index   │  │   (K clusters,   │ │  dict of lists,  │
    └─────────────────┘  │   soft probs)    │ │  cos-sim lookup) │
              ▲          └──────────────────┘ └──────────────────┘
              │                    │                   │
              └────────────────────┴───────────────────┘
                                   │
                          ┌────────▼────────┐
                          │   FastAPI        │
                          │   POST /query    │
                          │   GET  /cache/.. │
                          │   DELETE /cache  │
                          └─────────────────┘
```

---

## Design Decisions

### Embedding Model: `all-MiniLM-L6-v2`
384-dimensional embeddings — compact, fast (~5x faster than mpnet), and well-suited for semantic similarity on news-style text. L2-normalised so cosine similarity reduces to a dot product.

### Vector Store: ChromaDB
Pure Python, zero-infrastructure, persistent HNSW index. Supports metadata-filtered queries (by cluster, category). The right tool at this scale (~18k docs); FAISS would only pay off above ~1M vectors.

### Fuzzy Clustering: Gaussian Mixture Models (GMM)
GMM is the canonical probabilistic clustering algorithm — `predict_proba()` returns a full distribution over K clusters per document. PCA to 50 dims before GMM avoids the curse of dimensionality in 384D space. K is selected by BIC sweep over K ∈ [10, 30].

### Semantic Cache: Pure Python dict-of-lists (cluster-bucketed)
- **Data structure**: `dict[cluster_id → list[CacheEntry]]`
- **Lookup complexity**: O(n/K) vs O(n) naive scan
- **Zero dependencies**: plain Python + NumPy only
- **Similarity threshold τ**: configurable via `CACHE_SIMILARITY_THRESHOLD` (default 0.85)

---

## Quick Start

### Prerequisites
- Python 3.11+
- ~4GB disk space (model weights + ChromaDB)
- ~8GB RAM for embedding 18k documents

### 1. Create virtual environment

```bash
cd trademarkia-ai-task
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Copy environment config

```bash
cp .env.example .env
# Edit .env if you want to change model, thresholds, etc.
```

### 3. Set up the corpus (run once)

This downloads the dataset, cleans it, embeds all documents, fits the fuzzy clustering model, and stores everything in ChromaDB.

```bash
python scripts/setup_corpus.py
```

> ⏱ Expected time: ~10-20 min on CPU, ~2-3 min on GPU.

### 4. (Optional) Inspect clustering quality

```bash
python scripts/run_clustering.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### `POST /query`

Semantic search with cache lookup.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "gun control legislation debate"}'
```

**Response (cache miss):**
```json
{
  "query": "gun control legislation debate",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...",
  "dominant_cluster": 7
}
```

**Response (cache hit — rephrased query):**
```json
{
  "query": "firearms regulation political discussion",
  "cache_hit": true,
  "matched_query": "gun control legislation debate",
  "similarity_score": 0.912,
  "result": "...",
  "dominant_cluster": 7
}
```

### `GET /cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.85,
  "active_clusters": 8
}
```

### `DELETE /cache`

```bash
curl -X DELETE http://localhost:8000/cache
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

**Interactive API docs**: http://localhost:8000/docs

---

## Docker (Bonus)

### Prerequisites
Run `setup_corpus.py` on the host first to populate `./data/`.

### Build & run

```bash
docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 -v $(pwd)/data:/app/data trademarkia-semantic-search
```

### Or with Docker Compose

```bash
docker compose up --build
```

---

## The Semantic Cache — Threshold Exploration

The similarity threshold τ is the most consequential tunable in the system:

| τ value | Behaviour | Use case |
|---------|-----------|----------|
| 0.70 | Very aggressive — topically related queries hit | Approximate / exploratory systems |
| 0.80 | Lenient — synonyms and paraphrases reliably hit | FAQ systems |
| **0.85** | **Balanced — paraphrases hit, facet-changes miss** | **Default (general-purpose)** |
| 0.92 | Strict — near-identical phrasing only | High-precision systems |
| 0.98 | Exact — effectively a string match | No semantic benefit |

Configure via: `CACHE_SIMILARITY_THRESHOLD=0.85` in `.env`

---

## Project Structure

```
trademarkia-ai-task/
├── src/
│   ├── config.py                  # Central configuration
│   ├── data/preprocessor.py       # Corpus cleaning pipeline
│   ├── embeddings/embedder.py     # Sentence-transformer wrapper
│   ├── vectorstore/chroma_store.py # ChromaDB interface
│   ├── clustering/fuzzy_clusterer.py # GMM soft clustering
│   ├── cache/semantic_cache.py    # Hand-built cluster-aware cache
│   └── api/
│       ├── main.py                # FastAPI app + endpoints
│       └── models.py              # Pydantic schemas
├── scripts/
│   ├── setup_corpus.py            # Part 1 + 2 setup
│   └── run_clustering.py          # Cluster analysis
├── data/                          # ChromaDB + model files (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
