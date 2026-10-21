"""
05_kmeans.py - Clustering with K-Means (Day 103)

Clustering is UNSUPERVISED learning: we have no answer labels.
We just ask the computer to group similar points together.
K-Means is told how many groups (K) to look for, then:
  1. places K "center" points at random,
  2. assigns each point to its nearest center,
  3. moves each center to the middle of its points,
  4. repeats steps 2-3 until nothing changes.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files; no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from data import output_path


def main():
    # Make 300 points that naturally form 3 blobs. The labels are thrown away
    # ("_") because K-Means must find the groups without seeing the answers.
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # n_init = try 10 starts
    labels = kmeans.fit_predict(X)

    print("Cluster centers (the middle of each group):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"  cluster {i}: x={center[0]:.2f}, y={center[1]:.2f}")

    sizes = np.bincount(labels)  # how many points landed in each cluster
    print("\nCluster sizes:")
    for i, size in enumerate(sizes):
        print(f"  cluster {i}: {size} points")

    # --- Scatter plot: each cluster in its own color, centers marked with a big X ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25, alpha=0.8)
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=250,
        edgecolor="black",
        label="centers",
    )
    ax.set_title("K-Means with K=3")
    ax.legend()
    fig.savefig(output_path("kmeans.png"), dpi=120)
    plt.close(fig)

    # --- Elbow plot: try K = 1..6 and record "inertia" ---
    # Inertia = total distance from each point to its own cluster center.
    # Lower is tighter, but it always goes down as K goes up, so we look for a bend.
    k_values = list(range(1, 7))
    inertias = []
    print("\nInertia for each K (lower = tighter clusters):")
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(model.inertia_)
        print(f"  K={k}: inertia = {model.inertia_:.1f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow plot")
    fig.savefig(output_path("elbow.png"), dpi=120)
    plt.close(fig)

    # How to pick K from the elbow plot:
    # Look for the "elbow": the point where the line suddenly flattens out.
    # Before the elbow, adding a cluster helps a lot. After it, the gain is tiny.
    # Pick the K at the elbow. Here the data was built with 3 blobs, so the elbow should be at 3.
    print("\nChart saved to outputs/kmeans.png and outputs/elbow.png")
    print("Read the elbow: the drop in inertia shrinks a lot after K=3.")

    return {
        "topic": "K-Means clustering (Day 103)",
        "key_number": f"Cluster sizes {sizes.tolist()}; elbow near K=3",
        "takeaway": "K-Means finds groups without labels; the elbow plot hints at how many groups exist.",
    }


if __name__ == "__main__":
    main()
