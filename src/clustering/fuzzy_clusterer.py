import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from src.config import (
    CLUSTER_MODEL_PATH,
    PCA_MODEL_PATH,
    N_CLUSTERS,
    PCA_N_COMPONENTS,
)


class FuzzyClusterer:
    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        pca_components: int = PCA_N_COMPONENTS,
        cluster_model_path: str = CLUSTER_MODEL_PATH,
        pca_model_path: str = PCA_MODEL_PATH,
    ) -> None:
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.cluster_model_path = cluster_model_path
        self.pca_model_path = pca_model_path

        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
    # reducing dimension , pca
    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "PCA not fitted — call fit() or load() first."
        return self.pca.transform(embeddings)

    def fit(
        self,
        embeddings: np.ndarray,
        sweep: bool = True,
        k_range: range = range(10, 36),
        random_state: int = 42,
    ) -> "FuzzyClusterer":
        print(f"[FuzzyClusterer] Fitting PCA ({self.pca_components} components) …")
        self.pca = PCA(n_components=self.pca_components, random_state=random_state)
        reduced = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.cumsum()[-1]
        print(
            f"[FuzzyClusterer] PCA explains {explained * 100:.1f} % of variance."
        )

        if sweep:
            self.n_clusters = self._select_k(reduced, k_range, random_state)
        else:
            print(f"[FuzzyClusterer] Using fixed K={self.n_clusters}.")

        print(f"[FuzzyClusterer] Fitting GMM (K={self.n_clusters}) …")
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="diag",
            max_iter=200,
            n_init=3,
            random_state=random_state,
            verbose=0,
        )
        self.gmm.fit(reduced)
        print(
            f"[FuzzyClusterer] GMM converged: {self.gmm.converged_}, "
            f"lower_bound={self.gmm.lower_bound_:.4f}"
        )

        self._save()
        return self

    def _select_k(
        self,
        reduced: np.ndarray,
        k_range: range,
        random_state: int,
    ) -> int:
        print(
            f"\n{'K':>4}  {'BIC':>12}  {'AIC':>12}  {'Log-Lik':>12}"
        )
        print("-" * 46)

        bics, aics, log_liks = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in k_range:
                gmm_k = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    max_iter=150,
                    n_init=1,
                    random_state=random_state,
                )
                gmm_k.fit(reduced)
                bic = gmm_k.bic(reduced)
                aic = gmm_k.aic(reduced)
                ll = gmm_k.lower_bound_
                bics.append(bic)
                aics.append(aic)
                log_liks.append(ll)
                print(f"{k:>4}  {bic:>12.2f}  {aic:>12.2f}  {ll:>12.4f}")

        best_idx = int(np.argmin(bics))
        best_k = list(k_range)[best_idx]
        print(f"\n[FuzzyClusterer] Best K by BIC = {best_k}\n")
        return best_k

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.gmm is not None, "GMM not fitted."
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        reduced = self._reduce(embeddings)
        return self.gmm.predict_proba(reduced)

    def dominant_cluster(self, embedding: np.ndarray) -> int:
        probs = self.predict_proba(embedding)
        return int(np.argmax(probs[0]))

    def get_all_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        return self.predict_proba(embeddings)

    def _save(self) -> None:
        Path(self.pca_model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, self.pca_model_path)
        joblib.dump(self.gmm, self.cluster_model_path)
        print(
            f"[FuzzyClusterer] Models saved → "
            f"{self.pca_model_path}, {self.cluster_model_path}"
        )

    def load(self) -> "FuzzyClusterer":
        self.pca = joblib.load(self.pca_model_path)
        self.gmm = joblib.load(self.cluster_model_path)
        self.n_clusters = self.gmm.n_components
        print(
            f"[FuzzyClusterer] Loaded K={self.n_clusters} GMM from "
            f"{self.cluster_model_path}"
        )
        return self

    def is_fitted(self) -> bool:
        return self.pca is not None and self.gmm is not None

    @classmethod
    def from_disk(cls) -> "FuzzyClusterer":
        return cls().load()
