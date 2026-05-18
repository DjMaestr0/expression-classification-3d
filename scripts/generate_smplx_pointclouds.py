import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "smplx"))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

import smplx
from expression_anchors import sample_anchor_parameters
from pointnet_model import LABEL_TO_ID


MODEL_PATH = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "smplx_pcd")

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_REMAP = {
    "surprise": "surprised",
}


def mesh_to_pointcloud(vertices: np.ndarray, faces: np.ndarray, n_points: int = 1024) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def normalize(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.abs(centered).max()
    if scale < 1e-8:
        return centered
    return centered / scale


def main():
    parser = argparse.ArgumentParser(description="Generate SMPL-X face point clouds for domain-gap experiments.")
    parser.add_argument("--samples-per-emotion", type=int, default=50)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    model = smplx.create(
        MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=50,
    )

    records = []

    for emotion in EMOTIONS:
        public_label = EMOTION_REMAP.get(emotion, emotion)
        emotion_dir = os.path.join(args.output_root, public_label)
        os.makedirs(emotion_dir, exist_ok=True)

        for index in range(args.samples_per_emotion):
            betas = torch.randn(1, model.num_betas) * 0.18
            expression, jaw_pose = sample_anchor_parameters(
                model,
                emotion,
                expression_noise=0.02,
                jaw_noise=0.01,
            )

            with torch.no_grad():
                out = model(
                    betas=betas,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    global_orient=torch.zeros(1, 3, dtype=torch.float32),
                )

            vertices = out.vertices.detach().cpu().numpy().squeeze()
            points = normalize(mesh_to_pointcloud(vertices, model.faces, args.points))
            out_path = os.path.join(emotion_dir, f"{public_label}_{index:04d}.npy")
            np.save(out_path, points)

            records.append(
                {
                    "path": out_path,
                    "label": public_label,
                    "label_id": LABEL_TO_ID[public_label],
                    "source": "smplx",
                }
            )

    meta_path = os.path.join(args.output_root, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"Generated {len(records)} SMPL-X point clouds")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
