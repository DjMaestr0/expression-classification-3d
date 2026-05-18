import os
import sys
import argparse

import cv2
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(BASE_DIR, "smplx"))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

import smplx
from smplx.joint_names import JOINT_NAMES
from expression_anchors import EMOTION_ORDER, sample_anchor_parameters, ACTIVE_PROTOTYPE_PATH
from render_utils import fer_preprocess, photoreal_preprocess, render_mesh

MODEL_PATH = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "synthetic_dataset_fer")
JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
FACE_TRACK_POINTS = [
    "left_eye",
    "right_eye",
    "nose",
    "mouth_top",
    "mouth_bottom",
    "left_mouth_1",
    "right_mouth_1",
    "jaw",
]


ASYMMETRY_SIGMA = {
    "neutral":  0.00,
    "happy":    0.01,
    "sad":      0.02,
    "angry":    0.05,
    "surprise": 0.01,
    "fear":     0.03,
    "disgust":  0.08,   # asymmetric upper-lip raise is the core cue
}

SAMPLE_CONFIG = {
    "neutral":  {"expr_noise": 0.015, "jaw_noise": 0.006, "beta_sigma": 0.14},
    "happy":    {"expr_noise": 0.025, "jaw_noise": 0.012, "beta_sigma": 0.16},
    "sad":      {"expr_noise": 0.020, "jaw_noise": 0.008, "beta_sigma": 0.16},
    "angry":    {"expr_noise": 0.022, "jaw_noise": 0.009, "beta_sigma": 0.16},
    "surprise": {"expr_noise": 0.020, "jaw_noise": 0.012, "beta_sigma": 0.15},
    "fear":     {"expr_noise": 0.020, "jaw_noise": 0.010, "beta_sigma": 0.15},
    "disgust":  {"expr_noise": 0.018, "jaw_noise": 0.007, "beta_sigma": 0.15},
}

def apply_asymmetry(expression_tensor, emotion, num_expression_coeffs):

    sigma = ASYMMETRY_SIGMA.get(emotion, 0.02)
    if sigma == 0.0:
        return expression_tensor

    expr = expression_tensor.clone()
    # Perturb a random ~40% of coefficients independently on each side.
    n = num_expression_coeffs
    left_idx = np.random.choice(n, size=n // 2, replace=False)
    right_idx = np.setdiff1d(np.arange(n), left_idx)

    noise = np.zeros(n, dtype=np.float32)
    noise[left_idx]  += np.random.normal(0.0, sigma, size=len(left_idx)).astype(np.float32)
    noise[right_idx] += np.random.normal(0.0, sigma * 0.3, size=len(right_idx)).astype(np.float32)

    expr[0] += torch.tensor(noise, dtype=torch.float32)
    return expr


def sample_eye_pose():
    return (
        torch.zeros(1, 3, dtype=torch.float32),
        torch.zeros(1, 3, dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate FER-aligned SMPL-X synthetic facial-expression images."
    )
    parser.add_argument("--images-per-class", type=int, default=100)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--style", choices=["fer", "photo"], default="fer")
    parser.add_argument("--output-size", type=int, default=None)
    args = parser.parse_args()

    is_data_driven = "coma" in ACTIVE_PROTOTYPE_PATH.lower() or "emoca" in ACTIVE_PROTOTYPE_PATH.lower()
    if is_data_driven:
        print(f"Data-driven anchor mode ({ACTIVE_PROTOTYPE_PATH})")

    print("Project root:", BASE_DIR)
    print("Dataset path:", args.output_path)
    print("Loading SMPL-X model...")

    model = smplx.create(
        MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_expression_coeffs=50,   # full FLAME expression space
    )

    faces = model.faces
    print("Model loaded.")
    print("Generating FER-aligned dataset...\n")

    for emotion in EMOTION_ORDER:
        save_dir = os.path.join(args.output_path, emotion)
        os.makedirs(save_dir, exist_ok=True)
        print("Generating:", emotion)

        for i in range(args.images_per_class):
            cfg = SAMPLE_CONFIG[emotion]
            betas = torch.randn([1, model.num_betas]) * cfg["beta_sigma"]
            expression, jaw_pose = sample_anchor_parameters(
                model,
                emotion,
                expression_noise=cfg["expr_noise"],
                jaw_noise=cfg["jaw_noise"],
                raw_mode=is_data_driven,
            )

            # Apply per-emotion asymmetry before the forward pass.
            expression = apply_asymmetry(expression, emotion, model.num_expression_coeffs)
            leye_pose, reye_pose = sample_eye_pose()

            # Small random head pose — keep variation tight so the face
            # stays centred and the crop works reliably.
            global_orient = torch.tensor(
                [[
                    np.random.uniform(-0.015, 0.015),
                    np.random.uniform(-0.018, 0.018),
                    np.random.uniform(-0.015, 0.015),
                ]],
                dtype=torch.float32,
            )

            output = model(
                betas=betas,
                expression=expression,
                global_orient=global_orient,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
            )

            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
            face_points = {
                name: joints[JOINT_INDEX[name]]
                for name in FACE_TRACK_POINTS
                if name in JOINT_INDEX
            }

            # Randomise render parameters per image.
            cam_dist = np.random.uniform(0.289, 0.299)
            x_sh = np.random.uniform(0.006, 0.014)
            y_sh = np.random.uniform(0.240, 0.252)
            img_size = 256

            rendered = render_mesh(
                vertices,
                faces,
                image_size=img_size,
                light_intensity=np.random.uniform(2.3, 2.9) if args.style == "photo" else np.random.uniform(2.1, 2.8),
                camera_distance=cam_dist,
                x_shift=x_sh,
                y_shift=y_sh,
                material_style="photo" if args.style == "photo" else "fer",
            )

            if args.style == "photo":
                image = photoreal_preprocess(
                    rendered,
                    output_size=args.output_size or 224,
                    vertices=vertices,
                    face_points=face_points,
                    camera_distance=cam_dist,
                    x_shift=x_sh,
                    y_shift=y_sh,
                    image_size=img_size,
                )
            else:
                image = fer_preprocess(
                    rendered,
                    output_size=args.output_size or 48,
                    blur_sigma=np.random.uniform(0.2, 0.9),
                    contrast_scale=np.random.uniform(1.05, 1.2),
                    brightness_shift=np.random.uniform(-8.0, 8.0),
                    noise_std=np.random.uniform(2.0, 6.0),
                    vertices=vertices,
                    face_points=face_points,
                    camera_distance=cam_dist,
                    x_shift=x_sh,
                    y_shift=y_sh,
                    image_size=img_size,
                )

            filename = os.path.join(save_dir, f"{emotion}_{i:04d}.png")
            success = cv2.imwrite(filename, image)

            if success:
                print("Saved:", filename)
            else:
                print("FAILED:", filename)

    print("\nDataset generation finished.")


if __name__ == "__main__":
    main()
