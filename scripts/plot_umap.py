import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from scripts.pointnet_model import LABELS

EMOTION_COLORS = {
    "neutral": "#888888",
    "happy": "#FFD700",
    "sad": "#4169E1",
    "angry": "#FF4500",
    "surprised": "#FF69B4",
    "fear": "#9932CC",
    "disgust": "#228B22",
}
COLOR_LIST = [EMOTION_COLORS[l] for l in LABELS]


def main():
    parser = argparse.ArgumentParser(description="UMAP projection of PointNet features.")
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import umap

    all_data = []
    for fp in args.features:
        d = np.load(fp)
        all_data.append({"features": d["features"], "labels": d["labels"], "sources": d["sources"]})

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    first_emb = reducer.fit_transform(all_data[0]["features"])
    embeddings = [first_emb]
    for d in all_data[1:]:
        embeddings.append(reducer.transform(d["features"]))

    n_plots = len(args.features)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    for idx, (emb, tag) in enumerate(zip(embeddings, args.labels)):
        ax = axes[idx]
        for li, label_name in enumerate(LABELS):
            mask = all_data[idx]["labels"] == li
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[EMOTION_COLORS[label_name]], s=3, label=label_name, alpha=0.7)
        ax.set_title(tag)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == n_plots - 1:
            ax.legend(markerscale=4, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.tag}_umap_grid.png"), dpi=150)
    plt.close(fig)

    all_feats = np.concatenate([d["features"] for d in all_data])
    all_labels = np.concatenate([d["labels"] for d in all_data])
    all_sources = np.concatenate([d["sources"] for d in all_data])
    all_emb = np.concatenate(embeddings)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for li, label_name in enumerate(LABELS):
        for src_name, marker in [("coma", "o"), ("smplx", "X")]:
            mask = (all_labels == li) & (all_sources == src_name)
            if mask.sum() == 0:
                continue
            ax2.scatter(
                all_emb[mask, 0], all_emb[mask, 1],
                c=[EMOTION_COLORS[label_name]], marker=marker, s=8, alpha=0.6,
                label=f"{label_name}_{src_name}",
            )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(markerscale=2, fontsize=6, loc="best", ncol=2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, f"{args.tag}_umap_combined.png"), dpi=150)
    plt.close(fig2)

    print(f"UMAP plots saved: {args.tag}_umap_grid.png, {args.tag}_umap_combined.png")


if __name__ == "__main__":
    main()
