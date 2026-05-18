import json
import os
from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
RATIO_TRAIN, RATIO_VAL, RATIO_TEST = 0.70, 0.15, 0.15
LABELS = ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust"]


def load_records(meta_path, source):
    with open(meta_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        r["source"] = source
    return records


def stratified_split(records, test_size, val_size):
    label_ids = [r["label_id"] for r in records]
    train_idx, temp_idx = train_test_split(
        np.arange(len(records)),
        test_size=test_size + val_size,
        stratify=label_ids,
        random_state=SEED,
    )
    temp_records = [records[i] for i in temp_idx]
    temp_labels = [records[i]["label_id"] for i in temp_idx]
    val_size_adj = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        np.arange(len(temp_records)),
        test_size=1.0 - val_size_adj,
        stratify=temp_labels,
        random_state=SEED,
    )
    train = [records[i] for i in train_idx]
    val = [temp_records[i] for i in val_idx]
    test = [temp_records[i] for i in test_idx]
    return train, val, test


def save_split(records, path):
    out = []
    for r in records:
        out.append({
            "path": r["path"],
            "label": r["label"],
            "label_id": r["label_id"],
            "source": r["source"],
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    smplx_recs = load_records(os.path.join(BASE_DIR, "smplx_pcd", "metadata.json"), "smplx")
    coma_recs = load_records(os.path.join(BASE_DIR, "coma_pcd", "metadata.json"), "coma")

    # Mesh datasets
    smplx_mesh_recs = load_records(os.path.join(BASE_DIR, "smplx_face_mesh", "metadata.json"), "smplx_mesh")
    coma_mesh_recs = load_records(os.path.join(BASE_DIR, "coma_mesh", "metadata.json"), "coma_mesh")

    split_dir = os.path.join(BASE_DIR, "splits")

    # PCD splits
    smplx_train, smplx_val, smplx_test = stratified_split(
        smplx_recs, RATIO_TEST, RATIO_VAL
    )
    coma_train, coma_val, coma_test = stratified_split(
        coma_recs, RATIO_TEST, RATIO_VAL
    )

    # Mesh splits
    smplx_mesh_train, smplx_mesh_val, smplx_mesh_test = stratified_split(
        smplx_mesh_recs, RATIO_TEST, RATIO_VAL
    )
    coma_mesh_train, coma_mesh_val, coma_mesh_test = stratified_split(
        coma_mesh_recs, RATIO_TEST, RATIO_VAL
    )

    save_split(smplx_train, os.path.join(split_dir, "smplx_train.json"))
    save_split(smplx_val, os.path.join(split_dir, "smplx_val.json"))
    save_split(smplx_test, os.path.join(split_dir, "smplx_test.json"))
    save_split(coma_train, os.path.join(split_dir, "coma_train.json"))
    save_split(coma_val, os.path.join(split_dir, "coma_val.json"))
    save_split(coma_test, os.path.join(split_dir, "coma_test.json"))

    # Mesh splits
    save_split(smplx_mesh_train, os.path.join(split_dir, "smplx_mesh_train.json"))
    save_split(smplx_mesh_val, os.path.join(split_dir, "smplx_mesh_val.json"))
    save_split(smplx_mesh_test, os.path.join(split_dir, "smplx_mesh_test.json"))
    save_split(coma_mesh_train, os.path.join(split_dir, "coma_mesh_train.json"))
    save_split(coma_mesh_val, os.path.join(split_dir, "coma_mesh_val.json"))
    save_split(coma_mesh_test, os.path.join(split_dir, "coma_mesh_test.json"))

    combined_train = smplx_train + coma_train
    combined_val = smplx_val + coma_val
    combined_test = smplx_test + coma_test
    save_split(combined_train, os.path.join(split_dir, "combined_train.json"))
    save_split(combined_val, os.path.join(split_dir, "combined_val.json"))
    save_split(combined_test, os.path.join(split_dir, "combined_test.json"))



    print(f"{'emotion':<12} {'spx_tr':<8} {'spx_val':<8} {'spx_te':<8} {'coma_tr':<8} {'coma_val':<8} {'coma_te':<8} {'spx_m_tr':<8} {'spx_m_v':<8} {'spx_m_t':<8} {'com_m_tr':<8} {'com_m_v':<8} {'com_m_t':<8}")
    print("-" * 96)
    for label in LABELS:
        lid = LABELS.index(label)
        counts = {}
        for name, split in [
            ("spx_tr", smplx_train), ("spx_val", smplx_val), ("spx_te", smplx_test),
            ("coma_tr", coma_train), ("coma_val", coma_val), ("coma_te", coma_test),
            ("spx_m_tr", smplx_mesh_train), ("spx_m_v", smplx_mesh_val), ("spx_m_t", smplx_mesh_test),
            ("com_m_tr", coma_mesh_train), ("com_m_v", coma_mesh_val), ("com_m_t", coma_mesh_test),
        ]:
            counts[name] = sum(1 for r in split if r["label_id"] == lid)
        print(f"{label:<12} {counts['spx_tr']:<8} {counts['spx_val']:<8} {counts['spx_te']:<8} "
              f"{counts['coma_tr']:<8} {counts['coma_val']:<8} {counts['coma_te']:<8} "
              f"{counts['spx_m_tr']:<8} {counts['spx_m_v']:<8} {counts['spx_m_t']:<8} "
              f"{counts['com_m_tr']:<8} {counts['com_m_v']:<8} {counts['com_m_t']:<8}")

    print(f"\nTotal: {len(smplx_train) + len(coma_train)} pcd train, {len(smplx_val) + len(coma_val)} val, {len(smplx_test) + len(coma_test)} test")
    print(f"Total: {len(smplx_mesh_train) + len(coma_mesh_train)} mesh train, {len(smplx_mesh_val) + len(coma_mesh_val)} val, {len(smplx_mesh_test) + len(coma_mesh_test)} test")


if __name__ == "__main__":
    main()
