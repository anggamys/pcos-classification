import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from .attention import SelfAttention


class DenseNet121Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)

        feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.attention = SelfAttention(embed_dim=feature_dim, num_heads=8)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        features = features.unsqueeze(1)
        features = self.attention(features)
        features = features.squeeze(1)

        out = self.classifier(features)
        return out
