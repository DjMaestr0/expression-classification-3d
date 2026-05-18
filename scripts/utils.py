import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

# Constants 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS = ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)


# Helpers 
def normalize(vertices: np.ndarray) -> np.ndarray:
    """Center and scale vertices to [-1, 1]."""
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = np.abs(centered).max()
    return centered / scale if scale > 1e-8 else centered


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(records, device="cpu"):
    """Inverse-frequency class weights for imbalanced data."""
    cnt = Counter(r["label_id"] for r in records)
    total = sum(cnt.values())
    weights = [total / cnt[i] for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


# Dataset
class PointCloudDataset(Dataset):
    """Loads .npy point clouds/meshes from a metadata JSON file."""

    def __init__(self, meta_path: str, ravel: bool = False):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.ravel = ravel

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        pts = np.load(r["path"]).astype(np.float32)
        if self.ravel:
            pts = pts.ravel()
        return torch.from_numpy(pts), int(r["label_id"])


# Training loops
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(pts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pts.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            logits = model(pts)
            loss = criterion(logits, labels)
            total_loss += loss.item() * pts.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
