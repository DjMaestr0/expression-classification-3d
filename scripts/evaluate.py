import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from models import MLPNet, PointNet, FastPointNet
from utils import LABELS

MODEL_REGISTRY = {
    "mlp": MLPNet,
    "pointnet": PointNet,
    "fastpointnet": FastPointNet,
}


def load_model(model_name, checkpoint_path, device, input_dim=3072):
    if model_name == "mlp":
        model = MLPNet(input_dim=input_dim, return_features=True).to(device)
    else:
        model = MODEL_REGISTRY[model_name](return_features=True).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle legacy checkpoints that used "net"/"cls" instead of "backbone"/"head"
    if any(k.startswith("net.") for k in state_dict):
        state_dict = {k.replace("net.", "backbone.").replace("cls.", "head."): v
                      for k, v in state_dict.items()}
    if any(k.startswith("feat.") for k in state_dict):
        state_dict = {k.replace("cls.", "head."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="mlp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-meta", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--label", required=True, help="Prefix for output files.")
    parser.add_argument("--input-dim", type=int, default=3072, help="Only used for MLP.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device, args.input_dim)

    with open(args.test_meta, "r", encoding="utf-8") as f:
        records = json.load(f)

    ravel = args.model == "mlp"
    all_feats, all_labels, all_sources, all_preds = [], [], [], []

    with torch.no_grad():
        for r in records:
            pts = np.load(r["path"]).astype(np.float32)
            if ravel:
                pts = pts.ravel()
            pts_t = torch.tensor(pts).unsqueeze(0).to(device)
            logits, feats = model(pts_t)
            all_preds.append(logits.argmax(1).item())
            all_feats.append(feats.cpu().numpy()[0])
            all_labels.append(r["label_id"])
            all_sources.append(r.get("source", ""))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    prefix = os.path.join(args.out_dir, args.label)

    with open(f"{prefix}_metrics.txt", "w") as f:
        f.write(f"accuracy: {acc:.4f}\nmacro_f1: {macro_f1:.4f}\nweighted_f1: {weighted_f1:.4f}\n")

    with open(f"{prefix}_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    plot_confusion_matrix(cm, f"{prefix}_confusion.png")

    np.savez(f"{prefix}_features.npz",
             features=np.array(all_feats), labels=all_labels, sources=np.array(all_sources))

    print(f"accuracy: {acc:.4f}  macro_f1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
