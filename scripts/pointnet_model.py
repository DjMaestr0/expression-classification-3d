import numpy as np
import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, input_dim=3072, num_classes=7, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(),)
        self.cls = nn.Linear(128, num_classes)
    def forward(self, x):
        features = self.net(x)
        logits = self.cls(features)
        if self.return_features: return logits, features
        return logits

class PointNet(nn.Module):
    def __init__(self, num_classes=7, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),)
        self.cls = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),)
    def forward(self, x):
        x = x.transpose(2, 1); x = self.feat(x)
        features = x.max(dim=-1).values; logits = self.cls(features)
        if self.return_features: return logits, features
        return logits

class FastPointNet(nn.Module):
    def __init__(self, num_classes=7, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.feat = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),)
        self.cls = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes),)
    def forward(self, x):
        x = x.transpose(2, 1); x = self.feat(x)
        features = x.max(dim=-1).values; logits = self.cls(features)
        if self.return_features: return logits, features
        return logits

def extract_features(model, pts):
    x = pts.transpose(2, 1); x = model.feat(x)
    return x.max(dim=-1).values
