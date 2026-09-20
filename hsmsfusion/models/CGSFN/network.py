"""CGSFN adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/MSFLabX/CGSFN
Paper: Covariance-Guided and Direction-Aware Scan Attention for Hyperspectral
Image Fusion (CGSFN).

The two named contributions map onto these modules:
  - CG-CAM : the covariance-guided cross-modal path, i.e. ``CACW``/``IFW``
             (a full covariance over the flattened spatial axis drives a
             per-channel gate) combined with ``CSANet``/``Cross_Self_Attention``
             (bidirectional cross-modal attention).
  - DA-SAM : ``MSAA`` with ``SFA_Improved1``, a window-partitioned multi-scale
             attention that mixes 3/5/7 branches along both spatial axes.

HSMS adaptation notes
---------------------
Upstream hardcodes ``F.interpolate(..., scale_factor=8)`` for its x8 CAVE
protocol.  This port makes the upsampling factor a constructor argument so the
toolbox's x4 protocol can be expressed.

Upstream validates and tests through a sliding-window reconstruction with
``training_size=64`` and ``stride=32``.  The port keeps that protocol through
the toolbox's tiled inference (``tile_size: 64``, ``shave: 16``, so the step is
32).  Full-image inference is not usable here: the cross-modal attention at the
first encoder level is quadratic in the feature-map area, so a 256x256 Chikusei
test scene would need a 268M-entry attention map per head.

Removed relative to upstream, with forward numerics unchanged:
  - ``SCAM`` and the ``cro`` branch of ``Cross_block``.  ``cro=False`` at every
    call site, so it was unreachable, and ``SCAM.forward`` returned
    ``x_l + x_r`` without using any of the attention it computed.
  - ``Block``, ``DualT`` and ``ffn``, none of which ``Net`` references.
  - ``MSAA.channel_agg`` and ``SFA_Improved1.cov_x``, both constructed but
    never read in ``forward``.
  - ``SFA_Improved1.num_branches`` (unused) and ``Net.mid`` (the downstream
    channel ladder 64/96/128/256 and ``yz_mid`` are hardcoded, so any value
    other than 64 breaks the graph).
  - the ``einops`` dependency, the unused ``numpy`` import, and the
    ``__main__`` debug block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _heads(tensor: Tensor, heads: int) -> Tensor:
    """Split the last axis into ``heads`` and move them in front of the tokens."""
    batch, length, _ = tensor.shape
    return tensor.reshape(batch, length, heads, -1).permute(0, 2, 1, 3)


def window_partition(tensor: Tensor, window_size: int) -> Tensor:
    """[B, C, H, W] -> [B * n_win, C, win, win]."""
    batch, channels, height, width = tensor.shape
    if height % window_size or width % window_size:
        raise ValueError(
            f"SFA window partitioning needs spatial sizes divisible by "
            f"{window_size}, got {height}x{width}."
        )
    windows = tensor.view(
        batch, channels, height // window_size, window_size, width // window_size, window_size
    )
    windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous()
    return windows.view(-1, channels, window_size, window_size)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """[B * n_win, C, win, win] -> [B, C, H, W]."""
    batch = int(windows.shape[0] // (height * width / window_size / window_size))
    channels = windows.shape[1]
    tensor = windows.view(
        batch, height // window_size, width // window_size, channels, window_size, window_size
    )
    tensor = tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    return tensor.view(batch, channels, height, width)


class FeedForward(nn.Module):
    """Point-wise feed-forward block used by ``CSABlock``."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class CrossSelfAttention(nn.Module):
    """Bidirectional cross-modal attention with a residual over both inputs."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.5) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv_1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        heads = self.heads

        q1, k1, v1 = (_heads(t, heads) for t in self.to_qkv_1(x).chunk(3, dim=-1))
        q2, k2, v2 = (_heads(t, heads) for t in self.to_qkv_2(y).chunk(3, dim=-1))

        dots1 = torch.einsum("b h i d, b h j d -> b h i j", q2, k1) * self.scale
        out1 = torch.einsum("b h i j, b h j d -> b h i d", dots1.softmax(dim=-1), v1)

        dots2 = torch.einsum("b h i d, b h j d -> b h i j", q1, k2) * self.scale
        out2 = torch.einsum("b h i j, b h j d -> b h i d", dots2.softmax(dim=-1), v2)

        merged = out1.permute(0, 2, 1, 3).reshape(batch, tokens, -1) + out2.permute(
            0, 2, 1, 3
        ).reshape(batch, tokens, -1)
        return self.to_out(merged) + x + y


class CSABlock(nn.Module):
    """CG-CAM cross-modal block: 1x1 embedding, cross attention, feed-forward."""

    def __init__(self, in_c1: int, in_c2: int) -> None:
        super().__init__()
        self.Embedding1 = nn.Conv2d(in_c1, in_c1, (1, 1))
        self.Embedding2 = nn.Conv2d(in_c2, in_c1, (1, 1))
        self.atten = CrossSelfAttention(dim=in_c1, heads=4, dim_head=18)
        self.FFN = FeedForward(in_c1, in_c1 // 2, 0.0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        height = x.size(2)
        embedded_x = self.Embedding1(x).flatten(2).transpose(1, 2)
        embedded_y = self.Embedding2(y).flatten(2).transpose(1, 2)

        attention = self.atten(embedded_x, embedded_y)
        out = attention + self.FFN(attention)

        batch, _, channels = out.shape
        return out.transpose(1, 2).reshape(batch, channels, height, -1)


class CSANet(nn.Module):
    """Wrapper that projects both modalities and runs one ``CSABlock``."""

    def __init__(self, in_c1: int, in_c2: int, in_c3: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c1, in_c3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c2, in_c3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.CSABlock1 = CSABlock(in_c3, in_c3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c3, out_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = self.CSABlock1(self.conv1(x), self.conv2(y))
        return self.conv3(out)


class CACW(nn.Module):
    """Covariance-guided channel weight predictor.

    Flattens the spatial axes into samples, builds the normalised channel
    covariance, and regresses one weight per channel from the flattened matrix.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(num_features * num_features, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, num_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_samples, num_features = x.size()
        if num_features != self.num_features:
            raise ValueError(
                f"CACW expected {self.num_features} features, got {num_features}."
            )
        centered = x - x.mean(dim=1, keepdim=True)
        covariance = torch.bmm(centered.transpose(1, 2), centered) / (num_samples - 1)

        norms = torch.norm(centered, dim=1, keepdim=True)
        denominator = norms * norms.transpose(1, 2)
        normalized_cov = covariance / denominator.clamp(min=1e-8)

        flattened_cov = normalized_cov.view(batch_size, -1)
        return self.mlp(flattened_cov).unsqueeze(-1)


class IFW(nn.Module):
    """Channel re-weighting driven by the spatial covariance of the features."""

    def __init__(self, channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cacw = CACW(num_features=channels, hidden_dim=hidden_dim)

    def forward(self, feature: Tensor) -> Tensor:
        batch_size, channels, _, _ = feature.size()
        channel_weights = self.cacw(feature.view(batch_size, channels, -1).transpose(1, 2))
        return feature * channel_weights.unsqueeze(-1)


class DepthConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class NWI(nn.Module):
    """Direction-aware scan prior: row-wise plus column-wise average pooling."""

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]
        gap_w = F.adaptive_avg_pool2d(x, (1, width))
        gap_h = F.adaptive_avg_pool2d(x, (width, 1))
        return gap_w + gap_h


class SFA_Improved1(nn.Module):
    """Window-partitioned multi-scale scan attention (DA-SAM).

    Three dilated-free 3/5/7 branches are built; the first two form the
    attention map that re-weights the third, which is then combined with a
    depthwise "focus" response and restored to the full feature map.
    """

    def __init__(self, in_channels: int, window_size: int = 8) -> None:
        super().__init__()
        self.window_size = window_size
        alpter = window_size
        reduced = in_channels // alpter

        self.conv1x1_list = nn.ModuleList(
            [
                nn.Conv2d(in_channels, reduced, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, reduced, kernel_size=5, padding=2),
                nn.Conv2d(in_channels, reduced, kernel_size=7, padding=3),
            ]
        )
        self.nwi = NWI()
        self.depthwise_conv_focus = nn.Conv2d(
            reduced, reduced, kernel_size=3, stride=1, padding=1, groups=reduced, bias=False
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(reduced, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        branch1, branch2, branch3 = (conv(x) for conv in self.conv1x1_list)

        win1 = window_partition(branch1, self.window_size)
        win2 = window_partition(branch2, self.window_size)
        batch_win, channels_win, win_h, win_w = win1.shape

        win1_flat = win1.view(batch_win, channels_win, -1).transpose(1, 2)
        win2_flat = win2.view(batch_win, channels_win, -1)
        att_map = F.softmax(torch.bmm(win1_flat, win2_flat), dim=-1)

        branch3_win = window_partition(branch3, self.window_size)
        attended_flat = torch.bmm(branch3_win.view(batch_win, channels_win, -1), att_map)
        attended = window_reverse(
            attended_flat.view(batch_win, channels_win, win_h, win_w),
            self.window_size,
            height,
            width,
        )

        fused = self.nwi(attended) + self.depthwise_conv_focus(attended)
        return self.final_conv(fused)


class MSAA(nn.Module):
    """Multi-scale attention aggregation over depthwise 3/5/7 branches."""

    def __init__(self, in_channels: int, alpter: int = 8) -> None:
        super().__init__()
        self.spatial_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.multi_scale_convs = nn.ModuleList(
            [
                DepthConv(in_channels, in_channels // alpter, kernel_size=3, padding=1),
                DepthConv(in_channels, in_channels // alpter, kernel_size=5, padding=2),
                DepthConv(in_channels, in_channels // alpter, kernel_size=7, padding=3),
            ]
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels // alpter, in_channels // alpter, 1),
            nn.ReLU(),
            SFA_Improved1(in_channels=in_channels // alpter),
            nn.Conv2d(in_channels // alpter, 1, 1),
            nn.Sigmoid(),
        )
        self.spa_cov = nn.Conv2d(in_channels // alpter, in_channels, kernel_size=1)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        spatial_x = self.spatial_proj(x)
        multi_scale = sum(conv(spatial_x) for conv in self.multi_scale_convs)
        spatial_attn = self.spatial_conv(multi_scale)
        spatial_refined = self.spa_cov(multi_scale * spatial_attn)
        return self.fusion(spatial_refined) + x


class Cross_block(nn.Module):
    """Encoder stage: project both modalities and downsample both by 2."""

    def __init__(self, in_c1: int, in_c2: int, mid_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c1, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c2, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
        )

    def forward(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        return self.down1(self.conv1(y)), self.down2(self.conv2(z))


class UpDe(nn.Module):
    """Decoder stage: upsample the coarser branch and fuse the skip branch."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Conv2d(in_channels=out_c + out_c, out_channels=out_c, kernel_size=(3, 3), padding=1)

    def forward(self, infr: Tensor, latt: Tensor) -> Tensor:
        return self.conv(torch.cat((self.up(latt), infr), dim=1))


class CGSFN(nn.Module):
    """Covariance-guided scan fusion network.

    The trunk predicts a residual over the bicubic-upsampled LR HSI, so the
    final output is ``tail(...) + Y0``.
    """

    def __init__(self, hsi_channels: int, msi_channels: int, scale: int = 4) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)

        mid = 64
        self.intro1 = nn.Sequential(
            nn.Conv2d(in_channels=self.hsi_channels, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )
        self.intro2 = nn.Sequential(
            nn.Conv2d(in_channels=self.msi_channels, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )

        self.En1 = Cross_block(in_c1=mid, in_c2=mid, mid_c=64)
        self.En2 = Cross_block(in_c1=64, in_c2=64, mid_c=96)
        self.En3 = Cross_block(in_c1=96, in_c2=96, mid_c=128)

        self.ifw1 = IFW(channels=64)
        self.ifw2 = IFW(channels=96)
        self.ifw3 = IFW(channels=128)

        self.csa1 = CSANet(64, 64, 96, 96)
        self.csa2 = CSANet(96, 96, 128, 128)
        self.csa3 = CSANet(128, 128, 256, 256)

        self.covup1 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(1, 1), padding=0)
        self.covup2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(1, 1), padding=0)
        self.covup3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=0)

        self.msaa1 = MSAA(96)
        self.msaa2 = MSAA(128)
        self.msaa3 = MSAA(256)

        self.cDe1 = UpDe(in_c=256, out_c=128)
        self.cDe2 = UpDe(in_c=128, out_c=96)
        self.cDe3 = UpDe(in_c=96, out_c=64)

        self.yz_mid = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1)),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(64, self.hsi_channels, kernel_size=(1, 1)),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        )

    def _check_geometry(self, lr_hsi: Tensor, hr_msi: Tensor) -> None:
        """Reject inputs the trunk cannot consume, before the convolutions run.

        Two structural requirements come from the architecture rather than from
        a tuning choice.  The three encoder stages halve the feature map, and
        the SFA axis-attention in ``MSAA`` partitions 8x8 windows, so the HR
        side has to be divisible by 64.  ``NWI`` pools rows and columns into
        ``[B, C, 1, W]`` and ``[B, C, W, 1]`` and adds them, which only
        broadcasts when the feature map is square.
        """
        _, _, hr_height, hr_width = hr_msi.shape
        if hr_height != hr_width:
            raise ValueError(
                f"CGSFN needs a square HR input because NWI combines a "
                f"[B, C, 1, W] and a [B, C, W, 1] pooling, got "
                f"{hr_height}x{hr_width}."
            )
        if hr_height % 64:
            raise ValueError(
                f"CGSFN needs an HR side divisible by 64 (three stride-2 encoder "
                f"stages times the SFA window size of 8), got {hr_height}."
            )
        expected_lr = hr_height // self.scale
        _, _, lr_height, lr_width = lr_hsi.shape
        if (lr_height, lr_width) != (expected_lr, expected_lr):
            raise ValueError(
                f"LR HSI should be {expected_lr}x{expected_lr} for a "
                f"{hr_height}x{hr_height} HR input at scale {self.scale}, got "
                f"{lr_height}x{lr_width}."
            )

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        self._check_geometry(lr_hsi, hr_msi)

        y0 = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic", align_corners=False)

        y = self.intro1(y0)
        z = self.intro2(hr_msi)
        y1, z1 = self.En1(y, z)
        y2, z2 = self.En2(y1, z1)
        y3, z3 = self.En3(y2, z2)

        yz_csa1 = (
            self.csa1(self.ifw1(y1), self.ifw1(z1)) + self.covup1(y1) + self.covup1(z1)
        )
        yz_csa2 = (
            self.csa2(self.ifw2(y2), self.ifw2(z2)) + self.covup2(y2) + self.covup2(z2)
        )
        yz_csa3 = (
            self.csa3(self.ifw3(y3), self.ifw3(z3)) + self.covup3(y3) + self.covup3(z3)
        )

        yz_msaa1 = self.msaa1(yz_csa1)
        yz_msaa2 = self.msaa2(yz_csa2)
        yz_msaa3 = self.msaa3(yz_csa3)

        yz_mid = self.yz_mid(torch.cat((y, z), dim=1))

        yz_cde1 = self.cDe1(yz_msaa2, yz_msaa3)
        yz_cde2 = self.cDe2(yz_msaa1, yz_cde1)
        yz_cde3 = self.cDe3(yz_mid, yz_cde2)

        out = self.conv(torch.cat((self.up3(yz_cde1), yz_cde2), dim=1))
        return self.tail(torch.cat((self.up4(out), yz_cde3), dim=1)) + y0
