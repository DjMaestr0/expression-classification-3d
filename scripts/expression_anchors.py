
import json
import os

import numpy as np
import torch


EMOTION_ORDER = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprise",
    "fear",
    "disgust",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PRESET_PATH    = os.path.join(BASE_DIR, "config", "emotion_anchor_presets.json")
DEFAULT_PROTOTYPE_PATH = os.path.join(BASE_DIR, "config", "emotion_anchor_prototypes.json")
DEFAULT_CURATED_PATH   = os.path.join(BASE_DIR, "config", "curated_action_bank.json")
ACTIVE_PRESET_PATH     = os.environ.get("EMOTION_PRESET_PATH",    DEFAULT_PRESET_PATH)
ACTIVE_PROTOTYPE_PATH  = os.environ.get(
    "EMOTION_PROTOTYPE_PATH",
    DEFAULT_CURATED_PATH if os.path.exists(DEFAULT_CURATED_PATH) else DEFAULT_PROTOTYPE_PATH,
)


_GLOBAL_NOISE_SIGMA = 0.02
_TAIL_NOISE_SIGMA = 0.01

_PRIMARY_EXPR_SIGMA = {
    "neutral": 0.010,
    "happy": 0.035,
    "sad": 0.025,
    "angry": 0.030,
    "surprise": 0.028,
    "fear": 0.024,
    "disgust": 0.020,
}

_SIGNATURE_MIN = {
    "happy": {},
    "sad": {1: -1.4, 31: -1.2},
    "angry": {4: 1.00, 5: 0.85},
    "surprise": {0: 1.2},
    "fear": {0: 0.4, 1: 0.4},
    "disgust": {4: 0.95},
}

_SIGNATURE_MAX = {
    "neutral": {2: 0.20, 3: 0.15, 4: 0.10, 5: 0.08},
    "happy": {0: 0.80, 1: 0.80, 2: 0.55, 3: 0.45, 4: 0.18},
    "sad": {0: 0.10, 2: 0.20, 3: 0.15},
    "angry": {0: 0.20, 1: 0.20, 2: 0.12, 3: 0.10, 10: 0.08},
    "surprise": {4: 0.05, 5: 0.05},
    "fear": {2: 0.08, 3: 0.08},
    "disgust": {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10, 10: 0.03},
}

_JAW_LIMITS = {
    "neutral": (0.00, 0.01),
    "happy": (0.08, 0.16),
    "sad": (0.00, 0.10),
    "angry": (0.00, 0.03),
    "surprise": (0.22, 0.48),
    "fear": (0.10, 0.24),
    "disgust": (0.00, 0.02),
}


_JAW_NOISE = {
    "neutral":  0.010,
    "happy":    0.018,
    "sad":      0.010,
    "angry":    0.010,
    "surprise": 0.022,
    "fear":     0.016,
    "disgust":  0.008,
}


def load_anchor_preset(preset_path=DEFAULT_PRESET_PATH):
    with open(preset_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    version  = payload.get("version", 1)
    anchors  = {}
    targets  = {}
    au_noise = {}

    for emotion in EMOTION_ORDER:
        data = payload["emotions"][emotion]
        anchors[emotion] = np.array(data["anchor"], dtype=np.float32)
        targets[emotion] = data.get("targets", {})
        if version >= 2:
            raw = data.get("au_noise_ranges", {})
            au_noise[emotion] = {int(k): tuple(v) for k, v in raw.items()}
        else:
            au_noise[emotion] = {}

    n_coeffs = payload.get("n_coeffs", 10)
    return anchors, targets, au_noise, n_coeffs


ANCHOR_TABLE, TARGET_TABLE, AU_NOISE_TABLE, PRESET_N_COEFFS = load_anchor_preset(ACTIVE_PRESET_PATH)


def load_prototype_bank(prototype_path=ACTIVE_PROTOTYPE_PATH):
    if not os.path.exists(prototype_path):
        return {emotion: [ANCHOR_TABLE[emotion].copy()] for emotion in EMOTION_ORDER}

    with open(prototype_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    prototype_table = {}
    for emotion in EMOTION_ORDER:
        raw_entry = payload["emotions"][emotion]
        if isinstance(raw_entry, dict) and "variants" in raw_entry:
            variants = raw_entry["variants"]
            prototype_table[emotion] = [
                np.array(variant["anchor"], dtype=np.float32) for variant in variants
            ]
        elif isinstance(raw_entry, list):
            prototype_table[emotion] = [
                np.array(variant["anchor"], dtype=np.float32) for variant in raw_entry
            ]
        else:
            raise ValueError(f"Unsupported prototype format for emotion: {emotion}")
    return prototype_table


PROTOTYPE_TABLE = load_prototype_bank()


def sample_anchor_parameters(model, emotion, expression_noise=0.04, jaw_noise=None, raw_mode=False):
    prototypes = PROTOTYPE_TABLE.get(emotion, [ANCHOR_TABLE[emotion]])
    anchor     = prototypes[np.random.randint(len(prototypes))].copy()

    anchor_expr_len = max(0, len(anchor) - 3)
    n = min(model.num_expression_coeffs, PRESET_N_COEFFS, anchor_expr_len)
    expression = np.zeros(model.num_expression_coeffs, dtype=np.float32)

    expression[:n] = anchor[:n]

    if raw_mode:
        for i in range(model.num_expression_coeffs):
            expression[i] += float(np.random.normal(0.0, expression_noise))
    else:
        au_ranges    = AU_NOISE_TABLE.get(emotion, {})
        primary_sigma = _PRIMARY_EXPR_SIGMA.get(emotion, 0.02)
        global_sigma = _GLOBAL_NOISE_SIGMA if expression_noise == 0.04 else expression_noise
        tail_sigma = min(global_sigma, _TAIL_NOISE_SIGMA)

        for idx, (lo, hi) in au_ranges.items():
            if idx < model.num_expression_coeffs:
                expression[idx] += float(np.random.uniform(lo, hi))

        for i in range(n):
            if i not in au_ranges:
                expression[i] += float(np.random.normal(0.0, primary_sigma))

        for i in range(n, model.num_expression_coeffs):
            if i not in au_ranges:
                expression[i] += float(np.random.normal(0.0, tail_sigma))

        for idx, value in _SIGNATURE_MIN.get(emotion, {}).items():
            if idx < model.num_expression_coeffs:
                expression[idx] = max(expression[idx], value)
        for idx, value in _SIGNATURE_MAX.get(emotion, {}).items():
            if idx < model.num_expression_coeffs:
                expression[idx] = min(expression[idx], value)

    jaw_sigma = jaw_noise if jaw_noise is not None else _JAW_NOISE.get(emotion, 0.012)
    if len(anchor) >= anchor_expr_len + 3:
        jaw_pose = anchor[anchor_expr_len:anchor_expr_len + 3].copy()
    else:
        jaw_pose = anchor[-3:].copy()
    jaw_pose += np.random.normal(0.0, jaw_sigma, size=3).astype(np.float32)
    if not raw_mode:
        jaw_min, jaw_max = _JAW_LIMITS.get(emotion, (0.0, 0.2))
        jaw_pose[0] = np.clip(jaw_pose[0], jaw_min, jaw_max)
        jaw_pose  = np.clip(jaw_pose, [-0.05, -0.05, -0.05], [0.75, 0.05, 0.05])

    return (
        torch.tensor(expression, dtype=torch.float32).unsqueeze(0),
        torch.tensor(jaw_pose,   dtype=torch.float32).unsqueeze(0),
    )


def save_anchor_preset(anchor_table, target_table, preset_path, au_noise_table=None, n_coeffs=None):
    version = 2 if au_noise_table else 1
    payload = {
        "version":     version,
        "description": "FER-oriented SMPL-X emotion anchors.",
        "emotions":    {},
    }
    if n_coeffs is not None:
        payload["n_coeffs"] = int(n_coeffs)
    for emotion in EMOTION_ORDER:
        entry = {
            "anchor":  [float(v) for v in anchor_table[emotion]],
            "targets": target_table.get(emotion, {}),
        }
        if au_noise_table and emotion in au_noise_table:
            entry["au_noise_ranges"] = {
                str(k): list(v) for k, v in au_noise_table[emotion].items()
            }
        payload["emotions"][emotion] = entry

    with open(preset_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def set_prototype_table(prototype_table):
    global PROTOTYPE_TABLE
    PROTOTYPE_TABLE = prototype_table
