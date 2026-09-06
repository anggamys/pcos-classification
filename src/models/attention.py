import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """Multi-head Self-Attention (Artikel 1)"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.norm(x)
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention (Channel-wise)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excitation(w).view(B, C, 1, 1)
        return x * w.expand_as(x)


class CBAMAttention(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        w = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * w.expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.cat([avg_out, max_out], dim=1)
        w = self.sigmoid(self.conv(w))
        return x * w


class TransformerAttention(nn.Module):
    """Transformer-style Attention with Positional Encoding"""
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = self.norm(x)
        return x
