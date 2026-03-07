from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float = Field(description="Current cosine similarity threshold (τ)")
    active_clusters: int = Field(description="Number of cluster buckets with entries")


class HealthResponse(BaseModel):
    status: str
    corpus_size: int
    cache_entries: int
    n_clusters: int
    embedding_model: str
