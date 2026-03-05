import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


class CheXpertModel(nn.Module):
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()

        backbone = efficientnet_v2_m(weights=None)
        in_feats = backbone.classifier[1].in_features

        self.features = backbone.features
        self.pool = GeM()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_feats),
            nn.Dropout(dropout),
            nn.Linear(in_feats, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)