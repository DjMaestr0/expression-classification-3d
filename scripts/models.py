import torch.nn as nn


class MLPNet(nn.Module):
    """Simple MLP for raveled fixed-topology vertex arrays."""

    def __init__(self, input_dim: int = 3072, num_classes: int = 7, return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return (logits, features) if self.return_features else logits


class PointNet(nn.Module):
    """Standard PointNet for unordered point sets (N×3 input)."""

    def __init__(self, num_classes: int = 7, return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.transpose(2, 1)          # (B, 3, N)
        x = self.feat(x)
        features = x.max(dim=-1).values  # (B, 1024)
        logits = self.head(features)
        return (logits, features) if self.return_features else logits


class FastPointNet(nn.Module):
    """Lightweight PointNet variant — faster training on CPU."""

    def __init__(self, num_classes: int = 7, return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.feat = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(2, 1)          # (B, 3, N)
        x = self.feat(x)
        features = x.max(dim=-1).values  # (B, 128)
        logits = self.head(features)
        return (logits, features) if self.return_features else logits
