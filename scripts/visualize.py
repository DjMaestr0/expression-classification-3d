import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import BASE_DIR, LABELS

DEFAULT_OUT = os.path.join(BASE_DIR, "results", "figures")


def load_samples(data_dir, emotion, n=5):
    folder = os.path.join(data_dir, emotion)
    if not os.path.isdir(folder):
        return []
    files = sorted(os.listdir(folder))
    selected = random.sample(files, min(n, len(files)))
    samples = []
    for f in selected:
        d = np.load(os.path.join(folder, f)).astype(np.float32)
        if d.ndim == 1:
            d = d.reshape(-1, 3)
        samples.append(d)
    return samples


def plot_mesh(ax, verts, faces, color="steelblue"):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, shade=True, color=color, alpha=0.9, lw=0)


def plot_pcd(ax, pts):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="steelblue", s=1.5, alpha=0.9)


def set_view(ax):
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D face data.")
    parser.add_argument("--data-type", required=True, help="Directory name under project root.")
    parser.add_argument("--render-mode", choices=["mesh", "pcd"], required=True)
    parser.add_argument("--faces-file", default=None, help="Path to face_faces.npy (mesh mode).")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    random.seed(42)
    data_dir = os.path.join(BASE_DIR, args.data_type)
    os.makedirs(args.out_dir, exist_ok=True)

    faces = None
    if args.render_mode == "mesh":
        faces_path = args.faces_file or os.path.join(data_dir, "face_faces.npy")
        if os.path.exists(faces_path):
            faces = np.load(faces_path)
        else:
            print(f"Warning: faces file not found at {faces_path}, falling back to pcd mode")
            args.render_mode = "pcd"

    n_emotions = len(LABELS)
    fig, axes = plt.subplots(n_emotions, args.samples,
                             figsize=(args.samples * 2.5, n_emotions * 2.2),
                             subplot_kw={"projection": "3d"})
    for row, label in enumerate(LABELS):
        samples = load_samples(data_dir, label, args.samples)
        for col in range(args.samples):
            ax = axes[row, col] if n_emotions > 1 else axes[col]
            if col < len(samples):
                if args.render_mode == "mesh":
                    plot_mesh(ax, samples[col], faces)
                else:
                    plot_pcd(ax, samples[col])
                set_view(ax)
            else:
                ax.set_axis_off()
            if col == 0:
                ax.set_ylabel(label.capitalize(), fontsize=9, fontweight="bold", labelpad=8)

    title = f"{args.data_type} ({args.render_mode})"
    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{args.data_type}_{args.render_mode}_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
