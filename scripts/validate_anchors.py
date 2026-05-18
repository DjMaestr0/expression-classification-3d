import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import BASE_DIR

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config", "curated_combo_bank_coma.json")


def main():
    parser = argparse.ArgumentParser(description="Validate emotion anchor bank.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(args.config) as f:
        bank = json.load(f)

    print(f"Config: {args.config}")
    print(f"Version: {bank.get('version')} | N coeffs: {bank.get('n_coeffs')}")
    print()

    # Summary
    for emotion in EMOTIONS:
        variants = bank["emotions"].get(emotion, [])
        if not variants:
            print(f"  {emotion:12s}: NO VARIANTS")
            continue
        n = len(variants[0]["anchor"])
        print(f"  {emotion:12s}: {len(variants)} variant(s), anchor_len={n}")

    # Pairwise distances
    print("\nPairwise L2 distances (expression coeffs):")
    neutral_expr = np.array(bank["emotions"]["neutral"][0]["anchor"][:50])
    for emotion in EMOTIONS:
        variants = bank["emotions"].get(emotion, [])
        if not variants:
            continue
        expr = np.array(variants[0]["anchor"][:50])
        dist = np.linalg.norm(expr - neutral_expr)
        jaw = variants[0]["anchor"][50] if len(variants[0]["anchor"]) > 50 else 0
        print(f"  {emotion:12s}: L2_from_neutral={dist:.3f}  jaw={jaw:.3f}")

    # Critical pairs
    print("\nCritical pair separation:")
    pairs = [("angry", "disgust"), ("sad", "neutral"), ("surprise", "fear"),
             ("happy", "sad"), ("happy", "neutral")]
    all_ok = True
    for e1, e2 in pairs:
        if e1 not in bank["emotions"] or e2 not in bank["emotions"]:
            continue
        a1 = np.array(bank["emotions"][e1][0]["anchor"][:50])
        a2 = np.array(bank["emotions"][e2][0]["anchor"][:50])
        dist = np.linalg.norm(a1 - a2)
        status = "OK" if dist > 0.5 else "WARN"
        if dist <= 0.5:
            all_ok = False
        print(f"  {e1:12s} vs {e2:12s}: L2={dist:.3f}  [{status}]")

    print(f"\n{'All pairs pass.' if all_ok else 'Some pairs too close!'}")


if __name__ == "__main__":
    main()
