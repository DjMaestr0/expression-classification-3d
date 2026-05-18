import json
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
LABELS = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
NUM_CLASSES = 7
def normalize(vertices):
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = np.abs(centered).max()
    return centered / scale if scale > 1e-8 else centered
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
class PointCloudDataset(Dataset):
    def __init__(self, meta_path, ravel=False):
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.ravel = ravel
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        r = self.records[index]
        pts = np.load(r['path']).astype(np.float32)
        if self.ravel:
            pts = pts.ravel()
        return torch.from_numpy(pts), int(r['label_id'])
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for pts, labels in loader:
        pts = pts.to(device)
        labels = labels.to(device)
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
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(device)
            labels = labels.to(device)
            logits = model(pts)
            loss = criterion(logits, labels)
            total_loss += loss.item() * pts.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
