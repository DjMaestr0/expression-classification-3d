import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from expression_anchors import sample_anchor_parameters
from utils import BASE_DIR, LABEL_TO_ID, normalize

MODEL_PATH = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "smplx_face_mesh")
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_REMAP = {"surprise": "surprised"}
N_FACE_VERTS = 5023


def compute_face_indices(model):
    """Find 5023 face vertex indices via expression-direction magnitude + BFS growth."""
    expr_dirs = model.expr_dirs.detach().cpu().numpy()
    expr_mag = np.linalg.norm(expr_dirs, axis=1).max(axis=1)
    core_face = set(np.where(expr_mag > 1e-8)[0].tolist())

    # Build adjacency from mesh faces
    faces_np = model.faces_tensor.detach().cpu().numpy()
    adj = {}
    for f in faces_np:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))

    # Grow region to N_FACE_VERTS using connectivity priority
    face_set = set(core_face)
    frontier = {nb for v in core_face for nb in adj.get(v, set()) if nb not in face_set}

    while len(face_set) < N_FACE_VERTS and frontier:
        ranked = sorted(frontier, key=lambda v: -sum(1 for nb in adj.get(v, set()) if nb in face_set))
        to_add = min(len(ranked), N_FACE_VERTS - len(face_set))
        for v in ranked[:to_add]:
            face_set.add(v)
        frontier = {nb for v in face_set for nb in adj.get(v, set()) if nb not in face_set}

    return np.array(sorted(face_set)), faces_np


def main():
    parser = argparse.ArgumentParser(description="Generate SMPL-X face vertex arrays.")
    parser.add_argument("--samples-per-emotion", type=int, default=300)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boosted", action="store_true", help="Use boosted emotion anchors.")
    args = parser.parse_args()

    # Optionally activate boosted config
    if args.boosted:
        os.environ["EMOTION_PRESET_PATH"] = os.path.join(BASE_DIR, "config", "emotion_anchor_boosted.json")
        os.environ["EMOTION_PROTOTYPE_PATH"] = os.path.join(BASE_DIR, "config", "curated_action_bank_boosted.json")
        # Reload expression_anchors with new env
        import importlib
        import expression_anchors
        importlib.reload(expression_anchors)
        from expression_anchors import sample_anchor_parameters as _sap
    else:
        _sap = sample_anchor_parameters

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    import smplx
    model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral",
                         use_pca=False, num_betas=10, num_expression_coeffs=50)

    face_idx, faces_np = compute_face_indices(model)
    np.save(os.path.join(args.output_root, "face_indices.npy"), face_idx)
    print(f"Using {len(face_idx)} face vertex indices")

    # Save face topology for visualization
    face_set = set(face_idx.tolist())
    idx_to_local = {old: new for new, old in enumerate(face_idx)}
    local_faces = [[idx_to_local[int(v)] for v in f]
                   for f in faces_np if all(int(v) in face_set for v in f)]
    np.save(os.path.join(args.output_root, "face_faces.npy"), np.array(local_faces))
    print(f"Saved face topology ({len(local_faces)} triangles)")

    records = []
    for emotion in EMOTIONS:
        public_label = EMOTION_REMAP.get(emotion, emotion)
        emotion_dir = os.path.join(args.output_root, public_label)
        os.makedirs(emotion_dir, exist_ok=True)

        for i in range(args.samples_per_emotion):
            betas = torch.randn(1, model.num_betas) * 0.18
            expression, jaw_pose = _sap(model, emotion, expression_noise=0.02, jaw_noise=0.01)

            with torch.no_grad():
                out = model(betas=betas, expression=expression, jaw_pose=jaw_pose,
                            global_orient=torch.zeros(1, 3))

            verts = out.vertices.detach().cpu().numpy().squeeze()
            face_verts = normalize(verts[face_idx]).ravel().astype(np.float32)

            out_path = os.path.join(emotion_dir, f"{public_label}_{i:04d}.npy")
            np.save(out_path, face_verts)
            records.append({"path": out_path, "label": public_label,
                            "label_id": LABEL_TO_ID[public_label], "source": "smplx_face_mesh"})

    meta_path = os.path.join(args.output_root, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} SMPL-X face meshes → {meta_path}")


if __name__ == "__main__":
    main()
