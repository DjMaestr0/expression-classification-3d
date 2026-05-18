import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from scripts.pointnet_model import PointNet, FastPointNet

LABELS = ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-meta", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", choices=["pointnet", "fastpointnet"], default="pointnet")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "fastpointnet":
        model = FastPointNet(return_features=True).to(device)
    else:
        model = PointNet(return_features=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with open(args.test_meta, "r", encoding="utf-8") as f:
        records = json.load(f)

    all_feats, all_labels, all_sources, all_preds = [], [], [], []
    with torch.no_grad():
        for r in records:
            pts = torch.tensor(np.load(r["path"]).astype(np.float32)).unsqueeze(0).to(device)
            logits, feats = model(pts)
            pred = logits.argmax(1).item()
            all_feats.append(feats.cpu().numpy()[0])
            all_labels.append(r["label_id"])
            all_sources.append(r.get("source", ""))
            all_preds.append(pred)

    all_feats = np.array(all_feats)
    all_labels = np.array(all_labels)
    all_sources = np.array(all_sources)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    prefix = os.path.join(args.out_dir, args.label)
    with open(prefix + "_metrics.txt", "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"macro_f1: {macro_f1:.4f}\n")
        f.write(f"weighted_f1: {weighted_f1:.4f}\n")

    with open(prefix + "_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(prefix + "_confusion.png", dpi=150)
    plt.close(fig)

    np.savez(prefix + "_features.npz", features=all_feats, labels=all_labels, sources=all_sources)

    print(f"accuracy: {acc:.4f}  macro_f1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
