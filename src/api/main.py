from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.models import CacheStats, HealthResponse, QueryRequest, QueryResponse
from src.cache.semantic_cache import SemanticCache
from src.clustering.fuzzy_clusterer import FuzzyClusterer
from src.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CLUSTER_MODEL_PATH,
    EMBEDDING_MODEL,
    PCA_MODEL_PATH,
)
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_store import ChromaStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    print("\n[Startup] Initialising Trademarkia Semantic Search Service …")

    embedder = Embedder()
    app.state.embedder = embedder

    store = ChromaStore()
    app.state.store = store
    print(f"[Startup] ChromaDB corpus size: {store.count()} documents.")

    cluster_model_exists = (
        Path(CLUSTER_MODEL_PATH).exists() and Path(PCA_MODEL_PATH).exists()
    )
    if cluster_model_exists:
        clusterer = FuzzyClusterer.from_disk()
    else:
        raise RuntimeError(
            "Clustering models not found. "
            "Please run `python scripts/setup_corpus.py` first."
        )
    app.state.clusterer = clusterer
    print(f"[Startup] GMM loaded: K={clusterer.n_clusters} clusters.")

    cache = SemanticCache(threshold=CACHE_SIMILARITY_THRESHOLD)
    app.state.cache = cache
    print(f"[Startup] Semantic cache initialised (τ={CACHE_SIMILARITY_THRESHOLD}).")
    print("[Startup] Ready.\n")

    yield

    print("[Shutdown] Cleaning up …")


app = FastAPI(
    title="Trademarkia Semantic Search API",
    description=(
        "Lightweight semantic search over the 20 Newsgroups corpus with "
        "fuzzy clustering and a custom semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "Trademarkia Semantic Search API", "docs": "/docs"})


def _compute_result(
    query: str,
    embedding: np.ndarray,
    dominant_cluster: int,
    store: ChromaStore,
) -> str:
    try:
        result = store.query(
            query_embedding=embedding,
            n_results=1,
            where={"dominant_cluster": dominant_cluster},
        )
    except Exception:
        result = None

    if not result or not result["documents"][0]:
        result = store.query(query_embedding=embedding, n_results=1)

    if result and result["documents"][0]:
        doc = result["documents"][0][0]
        return doc[:500] + ("…" if len(doc) > 500 else "")

    return "No matching document found in corpus."


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with cache",
)
async def query_endpoint(request: QueryRequest):
    embedder: Embedder = app.state.embedder
    clusterer: FuzzyClusterer = app.state.clusterer
    store: ChromaStore = app.state.store
    cache: SemanticCache = app.state.cache

    embedding = embedder.encode_single(request.query)

    cluster_probs = clusterer.predict_proba(embedding)[0]
    dominant_cluster = int(np.argmax(cluster_probs))

    cached = cache.lookup(
        query=request.query,
        embedding=embedding,
        cluster_probs=cluster_probs,
    )
    if cached is not None:
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=cached.get("matched_query"),
            similarity_score=cached.get("similarity_score"),
            result=cached.get("result", ""),
            dominant_cluster=dominant_cluster,
        )

    result_text = _compute_result(
        query=request.query,
        embedding=embedding,
        dominant_cluster=dominant_cluster,
        store=store,
    )

    result_dict = {"result": result_text}
    cache.store(
        query=request.query,
        embedding=embedding,
        cluster_probs=cluster_probs,
        result=result_dict,
    )

    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_text,
        dominant_cluster=dominant_cluster,
    )


@app.get(
    "/cache/stats",
    response_model=CacheStats,
    summary="Current semantic cache statistics",
)
async def cache_stats():
    cache: SemanticCache = app.state.cache
    return CacheStats(**cache.stats)


@app.delete(
    "/cache",
    status_code=status.HTTP_200_OK,
    summary="Flush the semantic cache",
)
async def flush_cache():
    cache: SemanticCache = app.state.cache
    cache.flush()
    return {"message": "Cache flushed successfully.", "total_entries": 0}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health():
    return HealthResponse(
        status="ok",
        corpus_size=app.state.store.count(),
        cache_entries=len(app.state.cache),
        n_clusters=app.state.clusterer.n_clusters,
        embedding_model=EMBEDDING_MODEL,
    )
