import threading
import time
from typing import Optional

import numpy as np

from src.config import CACHE_SIMILARITY_THRESHOLD


class CacheEntry:
    __slots__ = ("query", "embedding", "result", "cluster_probs", "timestamp")

    def __init__(
        self,
        query: str,
        embedding: np.ndarray,
        result: dict,
        cluster_probs: np.ndarray,
    ) -> None:
        self.query = query
        self.embedding = embedding
        self.result = result
        self.cluster_probs = cluster_probs
        self.timestamp = time.time()


class SemanticCache:
    def __init__(self, threshold: float = CACHE_SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._buckets: dict[int, list[CacheEntry]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.RLock()

    def lookup(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_probs: np.ndarray,
    ) -> Optional[dict]:
        with self._lock:
            sorted_clusters = np.argsort(cluster_probs)[::-1] #Sort Clusters by Probability

            best_sim: float = 0.0
            best_entry: Optional[CacheEntry] = None

            for cluster_id in sorted_clusters[:2]:
                bucket = self._buckets.get(int(cluster_id), [])
                if not bucket:
                    continue

                stacked = np.stack([e.embedding for e in bucket], axis=0)
                sims = stacked @ embedding

                max_idx = int(np.argmax(sims))
                max_sim = float(sims[max_idx])

                if max_sim > best_sim:
                    best_sim = max_sim
                    best_entry = bucket[max_idx]

                if best_sim >= self.threshold:
                    break

            if best_sim >= self.threshold and best_entry is not None:
                self._hits += 1
                return {
                    **best_entry.result,
                    "cache_hit": True,
                    "matched_query": best_entry.query,
                    "similarity_score": round(best_sim, 6),
                }

            self._misses += 1
            return None

    def store(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_probs: np.ndarray,
        result: dict,
    ) -> None:
        dominant = int(np.argmax(cluster_probs))
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            result=result,
            cluster_probs=cluster_probs,
        )
        #Only one thread can access cache at a time
        with self._lock:
            if dominant not in self._buckets:
                self._buckets[dominant] = []
            self._buckets[dominant].append(entry)

    def flush(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "total_entries": sum(len(b) for b in self._buckets.values()),
                "hit_count": self._hits,
                "miss_count": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "threshold": self.threshold,
                "active_clusters": len(self._buckets),
            }

    def __len__(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"SemanticCache(entries={s['total_entries']}, "
            f"hit_rate={s['hit_rate']:.2%}, τ={self.threshold})"
        )
