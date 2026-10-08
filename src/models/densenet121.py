import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from .attention import CBAMAttention, SEAttention, SelfAttention, TransformerAttention


class DenseNet121Attention(nn.Module):
    """DenseNet-121 pretrained + modul attention + classifier biner PCOS.

    Args:
        num_classes (int): Jumlah kelas keluaran (default 1, logit biner).
        pretrained (bool): Muat bobot ImageNet bila True.
        dropout (float): Laju dropout classifier.
        attention_type (str): "self_attention", "se_net", "cbam",
            atau "transformer".
    """

    def __init__(
        self,
        pretrained=True,
        dropout=0.3,
        attention_type="self_attention",
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)

        feature_dim = self.backbone.classifier.in_features

        self.attention_type = attention_type

        if attention_type == "self_attention":
            self.attention = SelfAttention(embed_dim=feature_dim, num_heads=8)
            self.use_2d = False
        elif attention_type == "se_net":
            final_channels = 1024
            self.attention = SEAttention(channels=final_channels, reduction=16)
            self.use_2d = True
        elif attention_type == "cbam":
            final_channels = 1024
            self.attention = CBAMAttention(channels=final_channels, reduction=16)
            self.use_2d = True
        elif attention_type == "transformer":
            self.attention = TransformerAttention(
                embed_dim=feature_dim, num_heads=8, num_layers=2, dropout=dropout
            )
            self.use_2d = False
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """Forward: ekstraksi fitur, attention, lalu logit biner.

        Args:
            x (torch.Tensor): Batch citra (B, 3, H, W).

        Returns:
            torch.Tensor: Logit biner bentuk (B, 1).
        """
        features = self.backbone.features(x)

        if self.use_2d:
            features = self.attention(features)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        else:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)

            features = features.unsqueeze(1)
            features = self.attention(features)
            features = features.squeeze(1)

        out = self.classifier(features)
        return out
