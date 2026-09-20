"""SSIL-Net adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/Zhang-bd/SSIL-Net

The upstream network is a pan-sharpening model that consumes three inputs at
three resolutions of its own ladder, ``Model(h, m, p)``:

    pan  [B, p, H,    H]     a full-resolution panchromatic guide
    msi  [B, m, H/4,  H/4]   a coarser multispectral observation
    hsi  [B, h, H/16, H/16]  the coarsest hyperspectral observation

and predicts ``[B, h, H, H]`` as a residual over the bicubic-upsampled HSI.
Two paired upsamples of factor 4 compose the ladder, so upstream's total
HSI-to-output ratio is 16.

HSMS adaptation notes
---------------------
The toolbox's Chikusei protocol supplies two resolutions, not three:
``lr_hsi`` at H/scale and ``hr_msi`` at H, where the recipe's scale is 4.  The
ladder therefore collapses its top step, and every resolution factor is
rewritten to follow the recipe:

    lr_hsi (H/scale) --x scale--> the MSI/pan resolution (H) --x 1--> H

So ``up16`` becomes ``up scale``, and the MSI-side and residual-decoder side
factors become 1.  This is a change of scale factors only: no module is added
or removed, and not one channel width or block count differs from upstream
(``hidden`` stays 64, the three ``ResSPA`` blocks and four ``FuseBlock7``
blocks are untouched, and ``Multi_Scale`` keeps its structure).  Because the
single upstream ``up4`` module was reused at two *different* ratios upstream
(4 for the MSI path and 4 for the HSI path) but needs 1 and ``scale`` here, it
splits into ``up_msi_pan`` and ``up_hsi_msi``.  Both are parameterless
``nn.Upsample`` instances, so no weights are affected.

The three inputs are wired as:

    hsi <- lr_hsi                      [B, h, H/scale, H/scale]
    msi <- hr_msi                      [B, m, H,       H]
    pan <- hr_msi.mean(dim=1)          [B, 1, H,       H]

Every one of them comes from an observation the protocol already provides; the
panchromatic guide is a band-mean of the measured WV2 HRMSI, never a statistic
of the ground truth.  Upstream instead averages its 128-band reference cube,
which is not available here and would leak full-resolution target structure.

Removed relative to upstream, with forward numerics unchanged:
  - ``Freprocess2`` and ``SpeFE``, whole classes that ``Model`` never
    instantiates (their call sites are commented out upstream), so they held no
    parameters and no state-dict entries.
  - ``Model.up2`` and the ``ws`` bit of ``Multi_Scale``'s pooling attributes
    (``global_avg_pool``, ``global_max_pool``, ``max_pool``, ``avg_pool``),
    all parameterless and never called.
  - the ``einops`` dependency and the ``__main__`` profiling block.

Kept deliberately even though upstream never reads them: ``FuseBlock7.scale1``
and ``FuseBlock7.scale2``.  Their multiplications are commented out upstream,
so they receive no gradient, but dropping them would change the parameter
count and the state-dict layout away from upstream's.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _batch_heads(tensor: Tensor, heads: int) -> Tensor:
    """[B, C, H, W] -> [B, heads, C // heads, H * W]."""
    batch, channels, height, width = tensor.shape
    return tensor.reshape(batch, heads, channels // heads, height * width)


class Multi_Scale(nn.Module):
    """Cross-modal fusion block that mixes two same-resolution inputs."""

    def __init__(self, x_in_c: int, y_in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(x_in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(y_in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv11 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=out_c * 2, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_c * 2, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.conv1(x)
        y = self.conv2(y)

        x = self.conv_block(torch.cat((x, y), dim=1))

        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        y_avg = torch.mean(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)

        x = self.conv11(x) + torch.sigmoid(x_avg + x_max) * x
        y = self.conv22(y) + torch.sigmoid(y_avg + y_max) * y

        return self.conv(torch.cat((x, y), dim=1))


class ResSPA(nn.Module):
    """Multi-kernel spatial branch with a residual over the input."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        self.spa2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        self.spa3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels * 2, 3, 1, 1),
            nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        x1 = self.spa1(input)
        x2 = self.spa2(input)
        x3 = self.spa3(input)
        x = self.conv(torch.cat((x1, x2, x3), dim=1))
        return F.leaky_relu(x + input)


class Attention(nn.Module):
    """Transposed (channel-wise) attention with a learned temperature.

    ``q`` and ``k`` are split into ``num_heads`` groups of channels and the
    spatial axes are flattened, so the product is ``[B, heads, c, c]`` rather
    than a spatial ``[B, heads, HW, HW]`` map.  Cost is therefore linear in the
    number of pixels, which keeps whole-scene inference affordable.
    """

    def __init__(self, dim: int = 64, num_heads: int = 8, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias
        )
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch, _, height, width = x.shape

        key, value = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
        query = self.q_dwconv(self.q(x))

        query = _batch_heads(query, self.num_heads)
        key = _batch_heads(key, self.num_heads)
        value = _batch_heads(value, self.num_heads)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attn = (query @ key.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ value
        out = out.reshape(batch, -1, height, width)

        return self.project_out(out)


class FuseBlock7(nn.Module):
    """Spatial/spectral fusion stage fed by a shared transposed-attention pair."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.pointconv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.depthconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        # Declared by upstream and never read (its multiplications are commented
        # out).  Kept so the parameter count matches upstream.
        self.scale1 = nn.Parameter(1e-5 * torch.ones((1, channels, 1, 1)), requires_grad=True)
        self.scale2 = nn.Parameter(1e-5 * torch.ones((1, channels, 1, 1)), requires_grad=True)

    def forward(self, spa: Tensor, fre: Tensor) -> Tensor:
        fre = self.fre(fre)
        spa = self.spa(spa)
        # Both calls share one attention module, and the second consumes the
        # already-updated ``fre``; this is upstream's exact wiring.
        fre = self.fre_att(fre, spa) + fre
        spa = self.fre_att(spa, fre) + spa

        fuse = self.fuse(torch.cat((fre, spa), 1))

        spa = self.depthconv(fuse - spa)
        fre = self.pointconv(fuse - fre)
        return spa + fre + fuse


class SuperResolution(nn.Module):
    """Learned upsampler by an integer factor via a pixel-shuffle head."""

    def __init__(self, upscale_factor: int, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv4(x)
        return self.pixel_shuffle(x)


class SSILNet(nn.Module):
    """Spatial-spectral interaction network with a learned three-level ladder.

    ``scale`` is the recipe's HSI-to-output ratio.  The two ladder steps are
    ``scale`` (HSI to the MSI/pan resolution) and 1 (the MSI is already at full
    resolution), so their product equals ``scale``.
    """

    def __init__(self, hsi_channels: int, msi_channels: int, scale: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.pan_channels = 1
        self.hidden = int(hidden)
        scale = int(scale)
        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale}.")
        self.scale = scale

        # Ladder factors.  Upstream used 4 and 4 for a total of 16; the top step
        # collapses to 1 here because the protocol's HRMSI already sits at the
        # output resolution.
        self.hsi_to_msi = scale
        self.msi_to_pan = 1

        self.up_msi_pan = nn.Upsample(scale_factor=self.msi_to_pan, mode="bicubic", align_corners=True)
        self.up_hsi_msi = nn.Upsample(scale_factor=self.hsi_to_msi, mode="bicubic", align_corners=True)
        self.up_hsi_pan = nn.Upsample(scale_factor=self.hsi_to_msi, mode="bicubic", align_corners=True)

        self.srh4 = SuperResolution(self.hsi_to_msi, self.hsi_channels, self.hsi_channels)
        self.srm4 = SuperResolution(self.msi_to_pan, self.msi_channels, self.msi_channels)
        self.hidden4 = SuperResolution(self.msi_to_pan, self.hsi_channels, self.hsi_channels)

        self.mls1 = Multi_Scale(self.pan_channels, self.msi_channels, self.hidden)
        self.mls2 = Multi_Scale(self.msi_channels, self.hsi_channels, self.hsi_channels)
        self.mls3 = Multi_Scale(self.hidden, self.hidden, self.hidden)

        self.last_conv = nn.Sequential(
            nn.Conv2d(self.hidden, self.hidden * 2, kernel_size=3, padding=1),
            nn.Conv2d(self.hidden * 2, self.hsi_channels, kernel_size=3, padding=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.hidden * 2, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.hidden + self.pan_channels, self.hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.spa1 = ResSPA(self.hidden, self.hidden)
        self.spa2 = ResSPA(self.hidden, self.hidden)
        self.spa3 = ResSPA(self.hidden, self.hidden)

        self.fus1 = FuseBlock7(self.hidden)
        self.fus2 = FuseBlock7(self.hidden)
        self.fus3 = FuseBlock7(self.hidden)
        self.fus4 = FuseBlock7(self.hidden)

        self.hid = nn.Conv2d(self.hsi_channels, self.hidden, kernel_size=1)

        self.tt1 = nn.Sequential(
            nn.Conv2d(self.hidden * 4, self.hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden * 2, self.hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def _check_geometry(self, lr_hsi: Tensor, hr_msi: Tensor) -> None:
        """Reject inputs whose resolutions do not match the configured ladder."""
        _, _, hr_height, hr_width = hr_msi.shape
        if hr_height % self.scale or hr_width % self.scale:
            raise ValueError(
                f"HR input sides must be divisible by scale={self.scale} so that "
                f"the LR HSI is exactly HR/scale, got {hr_height}x{hr_width}."
            )
        _, _, lr_height, lr_width = lr_hsi.shape
        expected = (hr_height // self.scale, hr_width // self.scale)
        if (lr_height, lr_width) != expected:
            raise ValueError(
                f"LR HSI should be {expected[0]}x{expected[1]} for a "
                f"{hr_height}x{hr_width} HR input at scale {self.scale}, got "
                f"{lr_height}x{lr_width}."
            )

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        self._check_geometry(lr_hsi, hr_msi)

        # The panchromatic guide is a band-mean of the measured HRMSI.  It is an
        # observation, not a statistic of the reference cube.
        pan = hr_msi.mean(dim=1, keepdim=True)
        return self._fuse(pan, hr_msi, lr_hsi)

    def _fuse(self, pan: Tensor, msi: Tensor, hsi: Tensor) -> Tensor:
        hsr = self.srh4(hsi)
        msr = self.srm4(msi)

        # Residual baseline: bicubic upsample of the LR HSI to the output grid.
        h_out = self.up_hsi_pan(hsi)
        msr = self.up_msi_pan(msi) + msr
        hsi = self.up_hsi_msi(hsi) + hsr

        pm = self.mls1(pan, msr)
        mh = self.mls2(msi, hsi)
        mh = self.hidden4(mh) + h_out
        mh = self.hid(mh)

        x = self.mls3(pm, mh)

        y = self.conv1(torch.cat((x, mh), dim=1))
        x = self.conv0(torch.cat((x, pan), dim=1))

        x = self.spa1(x)
        x = self.spa2(x)
        x = self.spa3(x)

        s1 = self.fus1(x, y)
        s2 = self.fus2(x, s1)
        s3 = self.fus3(x, s2)
        s4 = self.fus4(x, s3)

        t1 = self.tt1(torch.cat((s1, s2, s3, s4), dim=1))
        return self.last_conv(t1) + h_out
