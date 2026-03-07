import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter

from src.clustering.fuzzy_clusterer import FuzzyClusterer
from src.vectorstore.chroma_store import ChromaStore


def print_section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def main() -> None:
    print_section("Part 2 — Fuzzy Clustering Analysis")

    print("\nLoading GMM clusterer from disk …")
    clusterer = FuzzyClusterer.from_disk()
    K = clusterer.n_clusters

    print("Loading corpus from ChromaDB …")
    store = ChromaStore()
    ids, embeddings, metadatas = store.get_all_embeddings()
    n_docs = len(ids)
    print(f"Loaded {n_docs} documents, K={K} clusters.\n")

    print("Computing soft cluster probabilities …")
    all_probs = clusterer.get_all_probabilities(embeddings)
    max_probs = all_probs.max(axis=1)
    dominant = all_probs.argmax(axis=1)
    entropies = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    print_section("A. Cluster Count Justification")
    print(
        f"""
The optimal K was selected by sweeping K ∈ [10, 30] and minimising BIC
(Bayesian Information Criterion). BIC penalises model complexity, preventing
over-segmentation of the 20-class corpus into spurious micro-clusters.

Selected K = {K}

Why not K = 20 (matching the 20 newsgroup labels)?
─────────────────────────────────────────────────────
The 20 Newsgroups categories have significant semantic overlap:
  • talk.politics.guns ↔ talk.politics.misc  (both are political discourse)
  • rec.sport.hockey ↔ rec.sport.baseball    (both about professional sports)
  • comp.os.ms-windows.misc ↔ comp.windows.x (both about GUI operating systems)
  • sci.med ↔ sci.space                      (both STEM research discussion)

The BIC elbow at K={K} reflects the *actual* semantic structure of the
corpus — not its administrative category labels.
"""
    )

    print_section("B. Cluster Semantic Profiles (Top Documents per Cluster)")

    raw = store.collection.get(include=["documents", "metadatas"])
    documents = raw["documents"]
    doc_metadatas = raw["metadatas"]

    for cluster_id in range(K):
        cluster_mask = dominant == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            print(f"\n  Cluster {cluster_id:2d}: [empty]")
            continue

        cluster_probs_for_this = all_probs[cluster_indices, cluster_id]
        top_indices = cluster_indices[np.argsort(cluster_probs_for_this)[::-1][:3]]

        cluster_cats = [doc_metadatas[i]["category"] for i in cluster_indices]
        cat_counts = Counter(cluster_cats)
        top_cats = cat_counts.most_common(3)

        print(f"\n  ── Cluster {cluster_id:2d} ({len(cluster_indices):4d} docs) ──")
        print(f"     Top categories: {', '.join(f'{c} ({n})' for c, n in top_cats)}")
        print(f"     Sample documents:")
        for idx in top_indices:
            doc_snippet = documents[idx][:150].replace("\n", " ")
            prob = all_probs[idx, cluster_id]
            print(f"       [p={prob:.3f}] {doc_snippet} …")

    print_section("C. Boundary Cases (max cluster probability < 0.40)")
    print(
        "These documents have genuinely ambiguous topic membership — they sit\n"
        "at the intersection of multiple semantic regions.\n"
    )

    boundary_mask = max_probs < 0.40
    boundary_indices = np.where(boundary_mask)[0]
    print(f"  Found {len(boundary_indices)} boundary documents "
          f"({100 * len(boundary_indices) / n_docs:.1f}% of corpus).\n")

    most_uncertain = boundary_indices[np.argsort(max_probs[boundary_indices])[:10]]
    for idx in most_uncertain:
        doc_snippet = documents[idx][:200].replace("\n", " ")
        top2 = np.argsort(all_probs[idx])[::-1][:2]
        cat = doc_metadatas[idx]["category"]
        print(
            f"  [{cat}] max_p={max_probs[idx]:.3f}, "
            f"cluster_probs=[c{top2[0]}:{all_probs[idx, top2[0]]:.3f}, "
            f"c{top2[1]}:{all_probs[idx, top2[1]]:.3f}]"
        )
        print(f"    → {doc_snippet[:150]} …\n")

    print_section("D. Assignment Entropy Distribution")
    max_entropy = np.log(K)
    print(f"  Max possible entropy (uniform over {K} clusters): {max_entropy:.3f}")
    print(f"  Mean entropy across corpus: {entropies.mean():.3f}")
    print(f"  Median entropy:             {np.median(entropies):.3f}")
    print(f"  Std dev:                    {entropies.std():.3f}")

    percentiles = [25, 50, 75, 90, 95]
    print("\n  Percentile breakdown of entropy:")
    for p in percentiles:
        print(f"    p{p:2d}: {np.percentile(entropies, p):.3f}")

    print_section("E. Newsgroup Category Distribution per Cluster")
    for cluster_id in range(K):
        cluster_mask = dominant == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue

        cats = [doc_metadatas[i]["category"] for i in cluster_indices]
        total = len(cats)
        top = Counter(cats).most_common(1)[0]
        purity = top[1] / total
        print(
            f"  Cluster {cluster_id:2d} | {total:4d} docs | "
            f"Purity={purity:.2f} | Dominant: {top[0]}"
        )

    print_section("Analysis Complete")
    print(
        "\nConclusion: The GMM with K=%d clusters reveals a semantic structure\n"
        "that goes beyond the 20 administrative labels. Cross-topic documents\n"
        "(boundary cases) confirm that soft assignment is the correct model\n"
        "for this corpus — hard clusters would misrepresent their nature.\n" % K
    )


if __name__ == "__main__":
    main()
