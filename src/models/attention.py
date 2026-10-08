import torch
from torch import nn


class SelfAttention(nn.Module):
    """Multi-head self-attention ala Artikel 1 (Q, K, V + proyeksi + norm).

    Args:
        embed_dim (int): Dimensi embedding fitur.
        num_heads (int): Jumlah attention head (default 8).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan self-attention multi-head pada sekuens token.

        Args:
            x (torch.Tensor): Token fitur bentuk (B, N, C).

        Returns:
            torch.Tensor: Fitur berbobot attention, bentuk sama.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.softmax(dim=-1)

        x = (attn_scores @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.norm(x)
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation: kalibrasi ulang bobot antar-kanal.

    Args:
        channels (int): Jumlah kanal feature map.
        reduction (int): Rasio reduksi bottleneck eksitasi (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kalikan feature map dengan bobot kanal hasil eksitasi.

        Args:
            x (torch.Tensor): Feature map bentuk (B, C, H, W).

        Returns:
            torch.Tensor: Feature map terbobot, bentuk sama.
        """
        B, C, _, _ = x.shape
        channel_weights = self.squeeze(x).view(B, C)
        channel_weights = self.excitation(channel_weights).view(B, C, 1, 1)

        return x * channel_weights.expand_as(x)


class CBAMAttention(nn.Module):
    """CBAM: attention kanal dilanjut attention spasial.

    Args:
        channels (int): Jumlah kanal feature map.
        reduction (int): Rasio reduksi gerbang kanal (default 16).
        kernel_size (int): Ukuran kernel gerbang spasial (default 7).
    """

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan gerbang kanal lalu gerbang spasial berurutan.

        Args:
            x (torch.Tensor): Feature map bentuk (B, C, H, W).

        Returns:
            torch.Tensor: Feature map terbobot, bentuk sama.
        """
        x = self.channel_gate(x)
        x = self.spatial_gate(x)

        return x


class ChannelGate(nn.Module):
    """Gerbang attention kanal dari pooled rata-rata + maksimum.

    Args:
        channels (int): Jumlah kanal feature map.
        reduction (int): Rasio reduksi lapisan fully-connected.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bobotkan tiap kanal dari statistik pooled rata-rata + maksimum.

        Args:
            x (torch.Tensor): Feature map bentuk (B, C, H, W).

        Returns:
            torch.Tensor: Feature map terbobot kanal, bentuk sama.
        """
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        channel_weights = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        return x * channel_weights.expand_as(x)


class SpatialGate(nn.Module):
    """Gerbang attention spasial dari peta rata-rata + maksimum per lokasi.

    Args:
        kernel_size (int): Ukuran kernel konvolusi gerbang (default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bobotkan tiap lokasi spasial dari peta agregat kanal.

        Args:
            x (torch.Tensor): Feature map bentuk (B, C, H, W).

        Returns:
            torch.Tensor: Feature map terbobot spasial, bentuk sama.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_weights = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(spatial_weights))

        return x * spatial_weights


class TransformerAttention(nn.Module):
    """Encoder transformer bertumpuk untuk dependensi global token fitur.

    Args:
        embed_dim (int): Dimensi embedding token.
        num_heads (int): Jumlah attention head (default 8).
        num_layers (int): Jumlah lapisan encoder (default 2).
        dropout (float): Laju dropout (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lewatkan token melalui encoder transformer + normalisasi.

        Args:
            x (torch.Tensor): Token fitur bentuk (B, N, C).

        Returns:
            torch.Tensor: Token hasil encoding, bentuk sama.
        """
        x = self.transformer(x)
        x = self.norm(x)

        return x
