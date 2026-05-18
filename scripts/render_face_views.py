import os
import sys
import random
import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE = r"C:\Users\Lenovo\Downloads\thesis_project"
OUT = os.path.join(BASE, "results", "face_views_pcd")
os.makedirs(OUT, exist_ok=True)
random.seed(42)

sys.path.insert(0, os.path.join(BASE, "scripts"))
from utils import LABELS

COMA_FACES = None
SMPLX_FACES = None
SMPLX_FACES = None

def _load_coma_faces():
    import trimesh
    m = trimesh.load(os.path.join(BASE, "coma", "data", "template.obj"))
    return np.array(m.faces)


def _list_samples(data_type, emotion, n=5):
    folder = os.path.join(BASE, data_type, emotion)
    files = sorted(os.listdir(folder))
    selected = random.sample(files, min(n, len(files)))
    return [np.load(os.path.join(folder, f)).astype(np.float32).reshape(-1, 3) for f in selected]


def _pcd(ax, verts):
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
               c="steelblue", s=1.5, alpha=0.9)

def _mesh(ax, verts):
    global COMA_FACES
    if COMA_FACES is None:
        COMA_FACES = _load_coma_faces()
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=COMA_FACES, shade=True,
                    color="steelblue", alpha=0.9, lw=0, antialiased=True)

def _smplx_mesh(ax, verts):
    global SMPLX_FACES
    if SMPLX_FACES is None:
        SMPLX_FACES = np.load(os.path.join(BASE, "smplx_face_mesh", "face_faces.npy"))
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=SMPLX_FACES, shade=True,
                    color="steelblue", alpha=0.9, lw=0, antialiased=True)


def _save(ax, path):
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def render_views(data_type, drawer, name, sample_count=5):
    for label in LABELS:
        samples = _list_samples(data_type, label, sample_count)
        sub = os.path.join(OUT, name, label)
        os.makedirs(sub, exist_ok=True)
        for idx, verts in enumerate(samples):
            for view_name, elev, azim in [("front", 0, -90), ("side", 0, 0), ("top", 90, -90)]:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection="3d")
                drawer(ax, verts)
                ax.view_init(elev=elev, azim=azim)
                _save(ax, os.path.join(sub, f"{idx+1:02d}_{view_name}.png"))
    print(f"  {name}: done")


if __name__ == "__main__":
    print("Rendering individual face views...")

    render_views("smplx_pcd", _pcd, "smplx_pcd")
    render_views("smplx_face_mesh", _smplx_mesh, "smplx_face_mesh")
    render_views("smplx_face_mesh", _pcd, "smplx_face_pcd")
    render_views("coma_mesh", _mesh, "coma_mesh")
    render_views("coma_pcd", _pcd, "coma_pcd")

    print(f"\nAll saved to {OUT}")
    print("SMPL-X face mesh now rendered using only main connected component (no eye islands)")
