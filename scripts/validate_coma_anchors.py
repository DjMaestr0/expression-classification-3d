import json
import sys
import os

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "scripts"))
from expression_anchors import EMOTION_ORDER

COMBO_BANK_PATH = os.path.join(BASE_DIR, "config", "curated_combo_bank_coma.json")

EMOTION_NAMES = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

with open(COMBO_BANK_PATH) as f:
    bank = json.load(f)

print(f"Version: {bank.get('version')}")
print(f"N coeffs: {bank.get('n_coeffs')}")
print(f"Source: {bank.get('source', 'unknown')}")
print()

n_anchors = sum(len(bank["emotions"][e]) for e in bank["emotions"])
print(f"Total variants: {n_anchors}")
print()

for emotion in EMOTION_NAMES:
    variants = bank["emotions"].get(emotion, [])
    if not variants:
        print(f"  {emotion:12s}: NO VARIANTS")
        continue
    anchor = variants[0]["anchor"]
    n = len(anchor)
    n_samples = variants[0].get("n_samples", "?")
    print(f"  {emotion:12s}: {len(variants)} variant(s)  "
          f"anchor_len={n}  n_samples={n_samples}")
    if n != 53:
        print(f"    [WARN] expected 53, got {n}")

print()
print("Anchor distances from neutral:")
neutral_expr = np.array(list(bank["emotions"]["neutral"][0]["anchor"][:50]))

for emotion in EMOTION_NAMES:
    variants = bank["emotions"].get(emotion, [])
    if not variants:
        continue
    anchor = variants[0]["anchor"]
    expr = np.array(anchor[:50])
    jaw = anchor[50]
    dist = np.linalg.norm(expr - neutral_expr)
    print(f"  {emotion:12s}: L2={dist:.3f}  jaw={jaw:.3f}")

print()
print("Critical pair distances:")
pairs = [
    ("angry", "disgust"),
    ("sad", "neutral"),
    ("happy", "neutral"),
    ("surprise", "neutral"),
    ("fear", "neutral"),
    ("happy", "sad"),
    ("surprise", "fear"),
]
for e1, e2 in pairs:
    a1 = np.array(bank["emotions"][e1][0]["anchor"][:50])
    a2 = np.array(bank["emotions"][e2][0]["anchor"][:50])
    dist = np.linalg.norm(a1 - a2)
    status = "OK" if dist > 1.0 else "WARN"
    print(f"  {e1:12s} vs {e2:12s}: L2={dist:.3f}  [{status}]")

expect = {"neutral": 0.005, "happy": 0.12, "sad": 0.05,
          "angry": 0.015, "surprise": 0.35, "fear": 0.17, "disgust": 0.01}

print()
print("Jaw validation:")
for emotion in EMOTION_NAMES:
    variants = bank["emotions"].get(emotion, [])
    if not variants:
        continue
    jaw = variants[0]["anchor"][50]
    expected = expect.get(emotion, 0.0)
    match = "OK" if abs(jaw - expected) < 0.001 else "MISMATCH"
    print(f"  {emotion:12s}: jaw={jaw:.3f}  expected={expected:.3f}  [{match}]")

print()
print("=== Summary ===")
all_ok = True
for e1, e2 in pairs:
    if e1 in bank["emotions"] and e2 in bank["emotions"]:
        a1 = np.array(bank["emotions"][e1][0]["anchor"][:50])
        a2 = np.array(bank["emotions"][e2][0]["anchor"][:50])
        if np.linalg.norm(a1 - a2) < 0.5:
            print(f"  FAIL: {e1} vs {e2} too close ({np.linalg.norm(a1-a2):.3f})")
            all_ok = False
if all_ok:
    print("  All anchor pairs pass separation check.")
