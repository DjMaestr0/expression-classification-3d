import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from utils import BASE_DIR, LABELS

SEED = 42
SPLIT_DIR = os.path.join(BASE_DIR, "splits")

# Dataset definitions: (metadata_path, source_tag, prefix)
DATASETS = {
    "smplx_pcd":       ("smplx_pcd/metadata.json",       "smplx",          "smplx"),
    "coma_pcd":        ("coma_pcd/metadata.json",        "coma",           "coma"),
    "smplx_face_mesh": ("smplx_face_mesh/metadata.json", "smplx_face_mesh", "smplx_face_mesh"),
    "coma_mesh":       ("coma_mesh/metadata.json",       "coma_mesh",       "coma_mesh"),
}


def load_records(meta_path, source):
    with open(meta_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        r["source"] = source
    return records


def stratified_split(records):
    """70/15/15 stratified split."""
    labels = [r["label_id"] for r in records]
    train_idx, temp_idx = train_test_split(
        np.arange(len(records)), test_size=0.30, stratify=labels, random_state=SEED)
    temp_labels = [records[i]["label_id"] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        np.arange(len(temp_idx)), test_size=0.5, stratify=temp_labels, random_state=SEED)
    return (
        [records[i] for i in train_idx],
        [records[temp_idx[i]] for i in val_idx],
        [records[temp_idx[i]] for i in test_idx],
    )


def save_split(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = [{"path": r["path"], "label": r["label"], "label_id": r["label_id"], "source": r["source"]}
           for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument("--only", choices=list(DATASETS.keys()), default=None,
                        help="Process only this dataset.")
    args = parser.parse_args()

    os.makedirs(SPLIT_DIR, exist_ok=True)
    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    all_splits = {}
    for name, (meta_rel, source, prefix) in targets.items():
        meta_path = os.path.join(BASE_DIR, meta_rel)
        if not os.path.exists(meta_path):
            print(f"  Skipping {name} (metadata not found: {meta_path})")
            continue

        records = load_records(meta_path, source)
        train, val, test = stratified_split(records)
        all_splits[name] = (train, val, test)

        save_split(train, os.path.join(SPLIT_DIR, f"{prefix}_train.json"))
        save_split(val, os.path.join(SPLIT_DIR, f"{prefix}_val.json"))
        save_split(test, os.path.join(SPLIT_DIR, f"{prefix}_test.json"))
        print(f"  {name}: train={len(train)} val={len(val)} test={len(test)}")

    # Combined PCD split (if both available)
    if "smplx_pcd" in all_splits and "coma_pcd" in all_splits:
        for split_name in ["train", "val", "test"]:
            idx = {"train": 0, "val": 1, "test": 2}[split_name]
            combined = all_splits["smplx_pcd"][idx] + all_splits["coma_pcd"][idx]
            save_split(combined, os.path.join(SPLIT_DIR, f"combined_{split_name}.json"))
        print("  combined: created")

    print("Done.")


if __name__ == "__main__":
    main()
