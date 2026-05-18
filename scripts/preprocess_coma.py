import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from utils import BASE_DIR, LABEL_TO_ID, normalize

DEFAULT_COMA_ROOT = os.path.join(BASE_DIR, "coma")

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


def find_mesh_paths(root: Path):
    """Yield (subject, expression_name, emotion_label, ply_path) for all valid CoMA files."""
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("FaceTalk_"):
            continue
        for expr_dir in sorted(subject_dir.iterdir()):
            if not expr_dir.is_dir():
                continue
            emotion = COMA_TO_EMOTION.get(expr_dir.name)
            if emotion is None:
                continue
            for ply_file in sorted(expr_dir.glob("*.ply")):
                yield subject_dir.name, expr_dir.name, emotion, ply_file


def process_mesh(ply_path: Path) -> np.ndarray:
    """Load mesh and return normalized raveled vertices."""
    mesh = trimesh.load(str(ply_path), process=False, force="mesh")
    return normalize(mesh.vertices.astype(np.float32)).ravel()


def process_pcd(ply_path: Path, n_points: int) -> np.ndarray:
    """Load mesh, sample surface points, normalize."""
    mesh = trimesh.load(str(ply_path), process=False, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return normalize(pts.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Preprocess CoMA data.")
    parser.add_argument("--mode", choices=["mesh", "pcd"], required=True)
    parser.add_argument("--coma-root", default=DEFAULT_COMA_ROOT)
    parser.add_argument("--output-root", default=None, help="Defaults to coma_mesh/ or coma_pcd/")
    parser.add_argument("--points", type=int, default=1024, help="Points per cloud (pcd mode).")
    parser.add_argument("--max-per-expression", type=int, default=None)
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = os.path.join(BASE_DIR, f"coma_{args.mode}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    expr_counts = {}

    for subject, expr_name, emotion, ply_file in find_mesh_paths(Path(args.coma_root)):
        key = (subject, expr_name)
        expr_counts[key] = expr_counts.get(key, 0) + 1
        if args.max_per_expression and expr_counts[key] > args.max_per_expression:
            continue

        if args.mode == "mesh":
            data = process_mesh(ply_file)
        else:
            data = process_pcd(ply_file, args.points)

        out_dir = output_root / emotion
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject}_{ply_file.stem}.npy"
        np.save(out_path, data)

        records.append({
            "path": str(out_path),
            "label": emotion,
            "label_id": LABEL_TO_ID[emotion],
            "source": f"coma_{args.mode}",
            "subject": subject,
            "expression": expr_name,
        })

    meta_path = output_root / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Processed {len(records)} CoMA samples ({args.mode}) → {meta_path}")


if __name__ == "__main__":
    main()
