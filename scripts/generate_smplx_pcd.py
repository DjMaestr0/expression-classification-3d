import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from expression_anchors import sample_anchor_parameters
from utils import BASE_DIR, LABEL_TO_ID, normalize

MODEL_PATH = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "smplx_pcd")
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_REMAP = {"surprise": "surprised"}


def main():
    parser = argparse.ArgumentParser(description="Generate SMPL-X face point clouds.")
    parser.add_argument("--samples-per-emotion", type=int, default=50)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    import smplx
    model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral",
                         use_pca=False, num_betas=10, num_expression_coeffs=50)

    records = []
    for emotion in EMOTIONS:
        public_label = EMOTION_REMAP.get(emotion, emotion)
        emotion_dir = os.path.join(args.output_root, public_label)
        os.makedirs(emotion_dir, exist_ok=True)

        for i in range(args.samples_per_emotion):
            betas = torch.randn(1, model.num_betas) * 0.18
            expression, jaw_pose = sample_anchor_parameters(model, emotion,
                                                            expression_noise=0.02, jaw_noise=0.01)
            with torch.no_grad():
                out = model(betas=betas, expression=expression, jaw_pose=jaw_pose,
                            global_orient=torch.zeros(1, 3))

            verts = out.vertices.detach().cpu().numpy().squeeze()
            mesh = trimesh.Trimesh(vertices=verts, faces=model.faces, process=False)
            pts, _ = trimesh.sample.sample_surface(mesh, args.points)
            pts = normalize(pts.astype(np.float32))

            out_path = os.path.join(emotion_dir, f"{public_label}_{i:04d}.npy")
            np.save(out_path, pts)
            records.append({"path": out_path, "label": public_label,
                            "label_id": LABEL_TO_ID[public_label], "source": "smplx"})

    meta_path = os.path.join(args.output_root, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} SMPL-X point clouds → {meta_path}")


if __name__ == "__main__":
    main()
