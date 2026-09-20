"""
Paper: SatFusion: A Unified Framework for Enhancing Remote Sensing Images
       via Multi-Frame and Multi-Source Images Fusion.
Venue: ACM MM 2026
DOI: https://doi.org/10.1145/3767308.3836355
Repository: https://github.com/yufeiTongZJU/SatFusion

HSMS adaptation notes
---------------------
The original SatFusion fuses multi-frame LRMS (RGB) with a single-channel PAN.
This port maps the same three-stage design onto HSMS inputs:

  MFIF  : LRHS [B, C_hsi, h, w]  -> HR semantic features via TRNet + PixelShuffle
  MSIF  : HRMS [B, C_msi, H, W]  -> spatial detail residual via PNN-style branch
  Fusion: residual composition + ColorCorrection

Channel counts follow the toolbox protocol (Chikusei: 128 HSI / 8 MSI).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared building blocks (ported from SatFusion misr_public_modules / pnn)
# ---------------------------------------------------------------------------


class DoubleConv2d(nn.Module):
    """Two-layer 2D conv block with PReLU and optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                bias=bias,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding="same",
                    bias=bias,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.PReLU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlockTR(nn.Module):
    def __init__(self, channel_size: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(channel_size, channel_size, kernel_size, padding=padding),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class EncoderTR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channel_size: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels, channel_size, kernel_size, padding=padding),
            nn.PReLU(),
        )
        self.res_layers = nn.Sequential(
            *[ResidualBlockTR(channel_size, kernel_size) for _ in range(num_layers)]
        )
        self.final = nn.Conv2d(channel_size, channel_size, kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.final(self.res_layers(self.init_layer(x)))


class DecoderTR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.final(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class SuperResTransformer(nn.Module):
    """TRNet SuperResTransformer with HRMS-guided CLS token (SatFusion*)."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        msi_channels: int = 8,
        use_msi_guide: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_msi_guide = use_msi_guide

        if use_msi_guide:
            self.msi_proj = nn.Conv2d(msi_channels, dim, kernel_size=1)
            self.msi_ln = nn.LayerNorm(dim)
            self.msi_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

    def forward(self, img: Tensor, msi_feat: Tensor | None) -> Tensor:
        # img: [B*H*W, R, dim]
        b, _, _ = img.shape
        if self.use_msi_guide and msi_feat is not None:
            # msi_feat: [B, C_msi, H, W] already spatially matching LR grid
            proj = self.msi_proj(msi_feat)  # [B, dim, H, W]
            flat = proj.permute(0, 2, 3, 1).reshape(-1, self.dim)
            cls_token = self.msi_mlp(self.msi_ln(flat)).unsqueeze(1)
        else:
            cls_token = repeat(self.cls_token, "() n d -> b n d", b=b)

        x = torch.cat((cls_token, img), dim=1)
        x = self.transformer(x)
        return x[:, 0]


class TRNet(nn.Module):
    """TRNet encoder / SuperResTransformer / decoder over spatial pixels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        kernel_size: int = 3,
        msi_channels: int = 8,
        use_msi_guide: bool = True,
    ) -> None:
        super().__init__()
        self.encode = EncoderTR(
            in_channels=in_channels,
            channel_size=dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.superres = SuperResTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            msi_channels=msi_channels,
            use_msi_guide=use_msi_guide,
        )
        self.decode = DecoderTR(in_channels=dim, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x: Tensor, msi_feat: Tensor | None = None) -> Tensor:
        # x: [B, C, H, W]
        batch_size, _, height, width = x.shape
        x = self.encode(x)  # [B, dim, H, W]
        # tokens: one per spatial location (R = 1 for single-frame HSMS)
        tokens = x.permute(0, 2, 3, 1).reshape(-1, 1, x.shape[1])  # [B*H*W, 1, dim]
        fused = self.superres(tokens, msi_feat=msi_feat)  # [B*H*W, dim]
        fused = fused.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        return self.decode(fused)


class PixelShuffleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zoom_factor: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels % (zoom_factor**2) != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by zoom_factor^2={zoom_factor**2}"
            )
        mid = in_channels // (zoom_factor**2)
        self.upsample = nn.Sequential(
            nn.PixelShuffle(zoom_factor),
            nn.Conv2d(mid, mid, kernel_size, padding="same", padding_mode="reflect"),
            nn.PReLU(),
            nn.Conv2d(mid, out_channels, kernel_size, padding="same", padding_mode="reflect"),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)


class ColorCorrection(nn.Module):
    """Residual 1x1 conv block for spectral balancing (SatFusion fusion composition)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x)


class PCNN(nn.Module):
    """PNN-style MSIF branch: injects HRMS spatial details as a residual."""

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        n1: int = 64,
        n2: int = 32,
        f1: int = 9,
        f2: int = 5,
        f3: int = 5,
    ) -> None:
        super().__init__()
        in_channels = hsi_channels + msi_channels
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, hsi_channels, kernel_size=f3, padding=f3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        x = torch.cat([ms, pan], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


# ---------------------------------------------------------------------------
# SatFusion (HSMS)
# ---------------------------------------------------------------------------


class SatFusion(nn.Module):
    """SatFusion* adapted for hyperspectral + multispectral fusion.

    Three stages follow the original paper:
      1. MFIF  — TRNet multi-source feature fusion + PixelShuffle upsampling
      2. MSIF  — PNN-style HRMS detail injection
      3. Fusion composition — residual sum + ColorCorrection
    """

    def __init__(
        self,
        hsi_channels: int = 128,
        msi_channels: int = 8,
        scale: int = 4,
        hidden_channels: int = 128,
        tr_dim: int = 64,
        tr_depth: int = 6,
        tr_heads: int = 8,
        tr_mlp_dim: int = 128,
        dropout: float = 0.1,
        encoder_layers: int = 2,
        pnn_n1: int = 64,
        pnn_n2: int = 32,
        use_msi_guide: bool = True,
    ) -> None:
        super().__init__()
        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels
        self.scale = scale

        # ---- Stage 1: MFIF ----
        # Encode LRHS into hidden features, fuse via TRNet, upsample to HR.
        self.mfif_encoder = DoubleConv2d(
            in_channels=hsi_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            dropout=0.0,
        )
        self.trnet = TRNet(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dim=tr_dim,
            depth=tr_depth,
            heads=tr_heads,
            mlp_dim=tr_mlp_dim,
            dropout=dropout,
            num_layers=encoder_layers,
            msi_channels=msi_channels,
            use_msi_guide=use_msi_guide,
        )
        self.sr = PixelShuffleBlock(
            in_channels=hidden_channels,
            out_channels=hsi_channels,
            zoom_factor=scale,
        )

        # ---- Stage 2: MSIF ----
        self.msif = PCNN(
            hsi_channels=hsi_channels,
            msi_channels=msi_channels,
            n1=pnn_n1,
            n2=pnn_n2,
        )

        # ---- Stage 3: Fusion composition ----
        self.color_correct = ColorCorrection(hsi_channels)

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        """Forward HSMS fusion.

        Args:
            lr_hsi: [B, C_hsi, h, w] low-resolution hyperspectral image.
            hr_msi: [B, C_msi, H, W] high-resolution multispectral image.

        Returns:
            [B, C_hsi, H, W] fused high-resolution hyperspectral image.
        """
        _, _, h, w = lr_hsi.shape
        _, _, height, width = hr_msi.shape
        expected_h, expected_w = h * self.scale, w * self.scale
        if (height, width) != (expected_h, expected_w):
            # Align HRMS to the exact super-resolved LR grid (defensive).
            hr_msi = F.interpolate(
                hr_msi, size=(expected_h, expected_w), mode="bilinear", align_corners=False
            )
            height, width = expected_h, expected_w

        # Stage 1: MFIF — encode LRHS, fuse with HRMS-guided transformer, upsample.
        feat = self.mfif_encoder(lr_hsi)  # [B, hidden, h, w]
        # Downsample HRMS guide to LR feature grid for token-level CLS guidance.
        msi_guide = F.interpolate(
            hr_msi, size=(h, w), mode="bilinear", align_corners=False
        )
        fused_feat = self.trnet(feat, msi_feat=msi_guide)  # [B, hidden, h, w]
        mfif_out = self.sr(fused_feat)  # [B, C_hsi, H, W]

        # Stage 2: MSIF — inject HRMS spatial details as residual.
        detail = self.msif(ms=mfif_out, pan=hr_msi)  # [B, C_hsi, H, W]

        # Stage 3: Fusion composition.
        out = mfif_out + detail
        out = self.color_correct(out)
        return out
