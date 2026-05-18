import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from .utils import (
    NUM_CLASSES, PointCloudDataset, set_seed,
    train_epoch, validate,
)
from .pointnet_model import PointNet, FastPointNet


def main():
    parser = argparse.ArgumentParser(description="Train PointNet on face point clouds.")
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--val-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["pointnet", "fastpointnet"], default="pointnet")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, seed: {args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)

    train_dataset = PointCloudDataset(args.train_meta, ravel=False)
    val_dataset = PointCloudDataset(args.val_meta, ravel=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if args.model == "fastpointnet":
        model = FastPointNet().to(device)
    else:
        model = PointNet().to(device)

    labels = [r["label_id"] for r in train_dataset.records]
    from collections import Counter
    cnt = Counter(labels)
    weights = [sum(cnt.values()) / cnt[i] for i in range(NUM_CLASSES)]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    csv_path = os.path.join(args.out_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                 f"{val_loss:.6f}", f"{val_acc:.6f}"]
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "best.pth"))

    torch.save(
        model.state_dict(),
        os.path.join(args.out_dir, "last.pth"))
    print(f"Done. Best val acc: {best_val_acc:.4f}. Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
