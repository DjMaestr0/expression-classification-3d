import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import MLPNet, PointNet, FastPointNet
from utils import NUM_CLASSES, PointCloudDataset, compute_class_weights, set_seed, train_epoch, validate


MODEL_REGISTRY = {
    "mlp": MLPNet,
    "pointnet": PointNet,
    "fastpointnet": FastPointNet,
}


def main():
    parser = argparse.ArgumentParser(description="Train expression classifier.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="mlp")
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--val-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=3072, help="Only used for MLP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Seed: {args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)

    # MLP uses raveled input; PointNet variants use (N, 3) tensors
    ravel = args.model == "mlp"
    train_ds = PointCloudDataset(args.train_meta, ravel=ravel)
    val_ds = PointCloudDataset(args.val_meta, ravel=ravel)

    train_loader = torch.utils.data.DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model
    if args.model == "mlp":
        model = MLPNet(input_dim=args.input_dim).to(device)
    else:
        model = MODEL_REGISTRY[args.model]().to(device)

    # Class-weighted loss
    weights = compute_class_weights(train_ds.records, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
            ])

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pth"))
    print(f"Done. Best val acc: {best_val_acc:.4f}. Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
