import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE = r"C:\Users\Lenovo\Downloads\thesis_project"
sys.path.insert(0, os.path.join(BASE, "scripts"))
from utils import LABELS

OUT = os.path.join(BASE, "results", "figures")
os.makedirs(OUT, exist_ok=True)
random.seed(42)

SMPLX_FACES = np.load(os.path.join(BASE, "smplx_face_mesh", "face_faces.npy"))
COMA_FACES = np.load(os.path.join(BASE, "smplx_face_mesh", "face_faces.npy"))

def load_coma_faces():
    import trimesh
    mesh = trimesh.load(os.path.join(BASE, "coma", "data", "template.obj"))
    return np.array(mesh.faces)

COMA_MESH_FACES = load_coma_faces()

def list_samples(data_type, emotion, n=5):
    folder = os.path.join(BASE, data_type, emotion)
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
                    triangles=faces, shade=True,
                    color=color, alpha=0.9, lw=0, antialiased=True)

def plot_pcd(ax, pts):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c="steelblue", s=1, alpha=0.8)

def set_view(ax):
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

def make_multi_sample_grid(data_type, title, faces, plot_fn, color="steelblue", sample_count=5):
    n_emotions = len(LABELS)
    fig, axes = plt.subplots(n_emotions, sample_count,
                             figsize=(sample_count * 2.5, n_emotions * 2.2),
                             subplot_kw={"projection": "3d"})
    for row, label in enumerate(LABELS):
        samples = list_samples(data_type, label, sample_count)
        for col in range(sample_count):
            ax = axes[row, col]
            if col < len(samples):
                if plot_fn.__name__ == "plot_mesh":
                    plot_fn(ax, samples[col], faces, color)
                else:
                    plot_fn(ax, samples[col])
                set_view(ax)
            if col == 0:
                ax.set_ylabel(label.capitalize(), fontsize=10, fontweight="bold", labelpad=10)
    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    safe = title.replace(" ", "_").replace(",", "").lower()
    path = os.path.join(OUT, f"{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

if __name__ == "__main__":
    print("Generating SMPL-X face grid (7 emotions x 5 samples)...")
    make_multi_sample_grid(
        "smplx_face_mesh", "SMPL-X Face Mesh (boosted expressions, 4x)",
        SMPLX_FACES, plot_mesh, "steelblue"
    )

    print("Generating COMA mesh grid (7 emotions x 5 samples)...")
    make_multi_sample_grid(
        "coma_mesh", "COMA Mesh (5023 fixed vertices)",
        COMA_MESH_FACES, plot_mesh, "steelblue"
    )

    print("Generating COMA point cloud grid (7 emotions x 5 samples)...")
    make_multi_sample_grid(
        "coma_pcd", "COMA Point Clouds (1024 random samples)",
        None, plot_pcd, None
    )

    print("Generating overlay per emotion (1 SMPL-X + 1 COMA mesh each)...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    for i, label in enumerate(LABELS):
        ax = axes[i]
        cma = list_samples("coma_mesh", label, 1)[0]
        smp = list_samples("smplx_face_mesh", label, 1)[0]
        ax.plot_trisurf(cma[:, 0], cma[:, 1], cma[:, 2],
                        triangles=COMA_MESH_FACES, shade=True,
                        color="steelblue", alpha=0.6, label="COMA")
        ax.plot_trisurf(smp[:, 0], smp[:, 1], smp[:, 2],
                        triangles=SMPLX_FACES, shade=True,
                        color="coral", alpha=0.6, label="SMPL-X")
        set_view(ax)
        ax.set_title(label.capitalize(), fontsize=10, fontweight="bold")
    axes[-1].axis("off")
    plt.suptitle("COMA vs SMPL-X Mesh Overlay (each emotion)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "overlay_all_emotions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved overlay_all_emotions.png")

    print(f"\nAll done! Figures in {OUT}")
