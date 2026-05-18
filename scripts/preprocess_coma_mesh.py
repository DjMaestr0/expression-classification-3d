import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from pointnet_model import LABEL_TO_ID


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_COMA_ROOT = os.path.join(BASE_DIR, "coma")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "coma_mesh")

COMA_TO_EMOTION = {
    "bareteeth": "happy",
    "high_smile": "happy",
    "mouth_up": "happy",
    "eyebrow": "surprised",
    "mouth_open": "surprised",
    "lips_back": "disgust",
    "lips_up": "disgust",
    "mouth_extreme": "angry",
    "mouth_side": "angry",
    "mouth_middle": "neutral",
    "mouth_down": "sad",
    "cheeks_in": "fear",
}


def load_mesh_vertices(ply_path: Path) -> np.ndarray:
    mesh = trimesh.load(str(ply_path), process=False, force="mesh")
    verts = mesh.vertices.astype(np.float32)
    verts -= verts.mean(axis=0, keepdims=True)
    scale = np.abs(verts).max()
    if scale > 1e-8:
        verts /= scale
    return verts.ravel()


def find_mesh_paths(root: Path):
    for subject_dir in root.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("FaceTalk_"):
            continue
        for expr_dir in subject_dir.iterdir():
            if not expr_dir.is_dir():
                continue
            emotion = COMA_TO_EMOTION.get(expr_dir.name)
            if emotion is None:
                continue
            for ply_file in expr_dir.glob("*.ply"):
                yield subject_dir.name, expr_dir.name, emotion, ply_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess CoMA meshes into raveled vertex arrays.")
    parser.add_argument("--coma-root", default=DEFAULT_COMA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-per-expression", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    expr_counts = {}

    for subject_name, expr_name, emotion, ply_file in find_mesh_paths(Path(args.coma_root)):
        key = (subject_name, expr_name)
        expr_counts[key] = expr_counts.get(key, 0) + 1
        if args.max_per_expression is not None and expr_counts[key] > args.max_per_expression:
            continue

        verts = load_mesh_vertices(ply_file)
        out_dir = output_root / emotion
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject_name}_{ply_file.stem}.npy"
        np.save(out_path, verts)

        records.append(
            {
                "path": str(out_path),
                "label": emotion,
                "label_id": LABEL_TO_ID[emotion],
                "source": "coma_mesh",
                "subject": subject_name,
                "expression": expr_name,
            }
        )

    meta_path = output_root / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"Processed {len(records)} CoMA meshes")
    print(f"Metadata: {meta_path}")
    print(f"Vertex dimension: {verts.shape[0]}")


if __name__ == "__main__":
    main()
