import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import LABELS

COLORS = {
    "neutral": "#888888",
    "happy": "#FFD700",
    "sad": "#4169E1",
    "angry": "#FF4500",
    "surprised": "#FF69B4",
    "fear": "#9932CC",
    "disgust": "#228B22",
}


def main():
    parser = argparse.ArgumentParser(description="UMAP projection of learned features.")
    parser.add_argument("--features", action="append", required=True, help="Path(s) to .npz files.")
    parser.add_argument("--labels", action="append", help="Display name for each feature file.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import umap

    datasets = [np.load(fp) for fp in args.features]
    display_names = args.labels or [os.path.basename(fp) for fp in args.features]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings = [reducer.fit_transform(datasets[0]["features"])]
    for d in datasets[1:]:
        embeddings.append(reducer.transform(d["features"]))

    # Grid plot
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for idx in range(n):
        ax = axes[idx]
        for li, label in enumerate(LABELS):
            mask = datasets[idx]["labels"] == li
            ax.scatter(embeddings[idx][mask, 0], embeddings[idx][mask, 1],
                       c=[COLORS[label]], s=3, label=label, alpha=0.7)
        ax.set_title(display_names[idx])
        ax.set_xticks([]); ax.set_yticks([])
        if idx == n - 1:
            ax.legend(markerscale=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.tag}_umap.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {args.tag}_umap.png")


if __name__ == "__main__":
    main()
