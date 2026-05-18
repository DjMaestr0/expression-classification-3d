import argparse
import json
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "smplx"))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

import smplx
from expression_anchors import sample_anchor_parameters
from utils import LABEL_TO_ID

MODEL_PATH = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "smplx_face_mesh")

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_REMAP = {"surprise": "surprised"}
N_FACE_VERTS = 5023


def normalize(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.abs(centered).max()
    if scale < 1e-8:
        return centered
    return centered / scale


def main():
    parser = argparse.ArgumentParser(description="Generate SMPL-X face vertex arrays.")
    parser.add_argument("--samples-per-emotion", type=int, default=300)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = smplx.create(
        MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=50,
    )

    # Compute correct face vertex indices (expression-influenced + grown to 5023)
    # Old code used verts[:5023] which included body parts — WRONG
    expr_dirs = model.expr_dirs.detach().cpu().numpy()
    expr_mag = np.linalg.norm(expr_dirs, axis=1).max(axis=1)
    core_face = set(np.where(expr_mag > 1e-8)[0].tolist())

    faces_tensor = model.faces_tensor.detach().cpu().numpy()
    edges = set()
    for f in faces_tensor:
        edges.add((int(f[0]), int(f[1])))
        edges.add((int(f[1]), int(f[2])))
        edges.add((int(f[0]), int(f[2])))
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    face_set = set(core_face)
    frontier = set()
    for v in core_face:
        for nb in adj.get(v, set()):
            if nb not in face_set:
                frontier.add(nb)
    while len(face_set) < N_FACE_VERTS and frontier:
        new_verts = sorted(frontier,
                          key=lambda v: -sum(1 for nb in adj.get(v, set()) if nb in face_set))
        to_add = min(len(new_verts), N_FACE_VERTS - len(face_set))
        for v in new_verts[:to_add]:
            face_set.add(v)
        frontier = set()
        for v in face_set:
            for nb in adj.get(v, set()):
                if nb not in face_set:
                    frontier.add(nb)

    face_idx = np.array(sorted(face_set))
    np.save(os.path.join(args.output_root, "face_indices.npy"), face_idx)
    print(f"Using {len(face_idx)} correct face vertex indices")

    records = []

    for emotion in EMOTIONS:
        public_label = EMOTION_REMAP.get(emotion, emotion)
        emotion_dir = os.path.join(args.output_root, public_label)
        os.makedirs(emotion_dir, exist_ok=True)

        for index in range(args.samples_per_emotion):
            betas = torch.randn(1, model.num_betas) * 0.18
            expression, jaw_pose = sample_anchor_parameters(
                model, emotion, expression_noise=0.02, jaw_noise=0.01,
            )

            with torch.no_grad():
                out = model(
                    betas=betas,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    global_orient=torch.zeros(1, 3, dtype=torch.float32),
                )

            verts = out.vertices.detach().cpu().numpy().squeeze()
            face_verts = normalize(verts[face_idx]).ravel().astype(np.float32)

            out_path = os.path.join(emotion_dir, f"{public_label}_{index:04d}.npy")
            np.save(out_path, face_verts)

            records.append({
                "path": out_path,
                "label": public_label,
                "label_id": LABEL_TO_ID[public_label],
                "source": "smplx_face_mesh",
            })

    meta_path = os.path.join(args.output_root, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    # Save face topology for visualization
    sub_faces = [f for f in faces_tensor if all(v in face_set for v in f)]
    local_faces = []
    idx_to_local = {old: new for new, old in enumerate(face_idx)}
    for f in sub_faces:
        local_faces.append([idx_to_local[int(v)] for v in f])
    np.save(os.path.join(args.output_root, "face_faces.npy"), np.array(local_faces))
    print(f"Saved face topology ({len(local_faces)} triangles)")

    print(f"Generated {len(records)} SMPL-X face meshes")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
