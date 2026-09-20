"""LSO adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/weili419/LSO
Paper: Solving Spatial-Spectral Fusion with Latent Spectral Operators
(ICML 2026).

Upstream calls the network ``LSOModel`` (aliased to ``Model``) and runs it as
``model(HR_MSI, LR_HSI)``:

    hr_msi  [B, m, H,      W     ]  the high-resolution multispectral observation
    lr_hsi  [B, h, H/sf,   W/sf  ]  the low-resolution hyperspectral observation

predicting ``[B, h, H, W]`` as a residual over the bicubic-upsampled HSI::

    lms   = bicubic(lr_hsi, sf)                       [B, h,     H, W]
    feat  = cat([hr_msi, lms], dim=1)                 [B, m + h, H, W]
    x     = spatial_encoder(feat)                     [B, g,     H, W]
    x     = cat([x, coord_grid], dim=1)               [B, g + 2, H, W]
    x1    = inc(x)                 # width            [B, w,     H,   W  ]
    x2    = down1(x1)              # width            [B, w,     H/2, W/2]
    x3    = down2(x2)              # width            [B, w,     H/4, W/4]
    y     = up1(process3(x3), process2(x2), hr_msi/2) + x2
    y     = up2(y,            process1(x1), hr_msi  ) + x1
    return lms + outc(y)

Each ``processN`` is a patchified ``NeuralSpectralBlock2d``: it lifts every
``patch_size`` window into ``num_token`` latent spectral tokens, expands them
onto ``num_basis`` sine/cosine modes, applies a learned ``width``-by-``2 *
num_basis`` mixing matrix, and decodes the tokens back onto the window.  The
``+2`` in ``inc`` is the normalized coordinate grid.

HSMS adaptation notes
---------------------
All three preceding ports had to negotiate a protocol mismatch; this one has
none.  Two observations at two resolutions, no panchromatic guide, no synthetic
intermediate rung, and a bicubic residual base is exactly what the toolbox
supplies, and ``sf`` is a free parameter upstream rather than a hardcoded
constant (the released CAVE/Harvard checkpoints cover x4 and x8).  So the port
is a change of surface, not of semantics:

  - ``n_bands``, ``n_select_bands`` and ``sf`` become ``hsi_channels``,
    ``msi_channels`` and ``scale``, and the argparse ``args`` namespace becomes
    explicit keyword arguments whose defaults are upstream's
    ``DEFAULT_LSO_ARGS``.
  - ``Up`` hardcodes ``in_channels + 3`` for the HR-MSI tensor it concatenates
    at every rung (``Up.forward(x1, x2, rgb)``).  The ``3`` is upstream's RGB
    band count, so it becomes ``msi_channels``; with Chikusei's 8-band WV2 MSI
    the three corresponding convolutions widen.  No module is added, removed or
    reordered.
  - ``forward`` takes ``(lr_hsi, hr_msi)``, the toolbox's argument order.

Kept deliberately even though upstream never runs them: ``MeanShift`` and its
two instances ``EDSR.sub_mean`` / ``EDSR.add_mean``.  Both call sites are
commented out in ``EDSR.forward``, so they hold frozen weights and receive no
gradient.  They are kept because the four released CAVE/Harvard checkpoints
contain ``spatial_encoder.sub_mean.*`` and ``spatial_encoder.add_mean.*``, and
upstream's ``load_state_dict`` override raises on any checkpoint key the model
lacks.  Verified against all four released checkpoints: with them present the
ported network reproduces every one of the checkpoint's 69 tensors with an
identical key and shape and strict-loads cleanly; without them those four keys
are the only difference.

Removed relative to upstream, with forward numerics unchanged:
  - ``Upsampler``, the EDSR ``url`` table, ``make_edsr`` and ``make_coord``.
    ``make_edsr_baseline`` always runs with ``no_upsampling=True`` here, so no
    tail is ever built and neither the upsampler nor the pretrained-weight URL
    table is reachable.
  - ``EDSR.load_state_dict``, an upstream override that silently tolerates
    mismatched ``tail.*`` keys when loading pretrained EDSR weights.  There is
    no tail to mismatch, and the toolbox loads Lightning checkpoints.
  - the ``bilinear`` argument of ``LSOModel`` and the ``bilinear=False`` branch
    of ``Up``: ``LSOModel`` accepts ``bilinear`` but never forwards it, so
    ``Up`` was only ever built with ``bilinear=True`` (the ``ConvTranspose2d``
    path was unreachable).
  - the ``numpy`` import, the commented-out debugging in ``forward``, the
    commented-out ``modes_list`` and PyPher leftovers, and the ``Model`` alias.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def default_conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    """EDSR's stride-1, same-size convolution."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True
    )


class MeanShift(nn.Conv2d):
    """EDSR's mean shift, kept for state-dict parity with the released weights.

    ``EDSR.forward`` has both of its call sites commented out upstream, so this
    never runs and its parameters stay frozen.  It is kept anyway because the
    four released CAVE/Harvard checkpoints carry
    ``spatial_encoder.sub_mean.*`` and ``spatial_encoder.add_mean.*``, and
    upstream's own ``load_state_dict`` override raises on any checkpoint key the
    model does not have.  Dropping it would make those checkpoints
    unstrict-loadable, so it stays.
    """

    def __init__(
        self,
        rgb_range: float,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """The EDSR residual block: two convolutions plus a scaled skip."""

    def __init__(
        self,
        conv,
        n_feats: int,
        kernel_size: int,
        act: nn.Module,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        body = [conv(n_feats, n_feats, kernel_size), act, conv(n_feats, n_feats, kernel_size)]
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x).mul(self.res_scale)
        return res + x


class EDSR(nn.Module):
    """EDSR's head and body, used here as a fixed-depth spatial encoder.

    ``make_edsr_baseline`` is called with ``no_upsampling=True`` and
    ``res_scale=1``, so this is upstream's ``EDSR`` with only the tail module
    dropped, which upstream never builds under those settings either.
    """

    def __init__(
        self,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 1.0,
        n_colors: int = 1,
        rgb_range: float = 1.0,
    ) -> None:
        super().__init__()
        kernel_size = 3
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(default_conv(n_colors, n_feats, kernel_size))
        body = [
            ResBlock(
                default_conv,
                n_feats,
                kernel_size,
                act=nn.ReLU(True),
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        body.append(default_conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

    def forward(self, x: Tensor) -> Tensor:
        # Both mean-shift call sites are commented out upstream, and are left out
        # here too so the forward pass matches.
        x = self.head(x)
        res = self.body(x)
        res = res + x
        return res


def make_edsr_baseline(
    n_resblocks: int = 16,
    n_feats: int = 64,
    res_scale: float = 1.0,
    n_colors: int = 1,
    rgb_range: float = 1.0,
) -> EDSR:
    """Build the baseline EDSR variant, matching upstream's defaults."""
    return EDSR(
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        res_scale=res_scale,
        n_colors=n_colors,
        rgb_range=rgb_range,
    )


class DoubleConv(nn.Module):
    """(convolution => LeakyReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """One multiscale rung down: a bilinear halving plus a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=1 / 2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class Up(nn.Module):
    """One multiscale rung up, re-injecting the rung's features and the MSI.

    ``msi_channels`` is the band count of the HR-MSI tensor concatenated at
    every rung; upstream spells it as a literal ``3`` inside
    ``in_channels + 3``.
    """

    def __init__(self, in_channels: int, out_channels: int, msi_channels: int) -> None:
        super().__init__()
        in_channels = in_channels + msi_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: Tensor, x2: Tensor, msi: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x1, x2, msi], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """The head that maps the decoder features back to HSI bands."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class NeuralSpectralBlock2d(nn.Module):
    """The patchified neural spectral block.

    Each ``patch_size`` window is lifted into ``num_token`` latent spectral
    tokens by an encoder attention, expanded onto ``2 * num_basis`` sine/cosine
    modes, mixed by a learned ``width``-by-``2 * num_basis`` matrix, and decoded
    back onto the window by a decoder attention.
    """

    def __init__(
        self,
        width: int,
        num_basis: int,
        patch_size: tuple[int, int] = (3, 3),
        num_token: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis
        self.head = 8
        if width % self.head:
            raise ValueError(
                f"The latent tokens split ``width`` over {self.head} heads, so "
                f"width must be a multiple of {self.head}, got {width}."
            )
        self.num_token = num_token

        # The spectral modes: 1, 2, ..., num_basis, scaled by pi in get_basis.
        self.register_buffer("modes_list", torch.arange(1, num_basis + 1, dtype=torch.float))
        self.weights = nn.Parameter((1 / width) * torch.rand(width, num_basis * 2, dtype=torch.float))
        self.latent = nn.Parameter(
            (1 / width) * torch.rand(self.head, num_token, width // self.head, dtype=torch.float)
        )
        self.encoder_attn = nn.Conv2d(width, width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv2d(width, width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        d = q.shape[-1]
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k) / math.sqrt(d))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x: Tensor) -> Tensor:
        """Attend from the learned latent tokens onto every patch."""
        batch, channels, _, _ = x.shape
        length = x.shape[2] * x.shape[3]
        latent_token = self.latent[None, :, :, :].repeat(batch, 1, 1, 1)
        x_tmp = (
            self.encoder_attn(x)
            .reshape(batch, channels * 2, -1)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(batch, length, self.head, channels // self.head, 2)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
        )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        return latent_token.permute(0, 1, 3, 2).contiguous().reshape(batch, channels, self.num_token)

    def latent_decoder_attn(self, x: Tensor, latent_token: Tensor) -> Tensor:
        """Attend from every patch back onto the latent tokens."""
        x_init = x
        batch, channels, height, width = x.shape
        length = height * width
        latent_token = (
            latent_token.reshape(batch, self.head, channels // self.head, self.num_token)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_tmp = (
            self.decoder_attn(x)
            .reshape(batch, channels, -1)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(batch, length, self.head, channels // self.head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.self_attn(x_tmp, latent_token, latent_token)
        return (
            x.permute(0, 1, 3, 2).contiguous().reshape(batch, channels, height, width) + x_init
        )

    def get_basis(self, x: Tensor) -> Tensor:
        """Expand ``[B, C, N]`` angles onto ``num_basis`` sine/cosine modes."""
        modes = self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        return torch.cat([torch.sin(modes), torch.cos(modes)], dim=-1)

    def compl_mul2d(self, inputs: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bilm,im->bil", inputs, weights)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        # Window edges that do not tile the feature map are zero-padded, then cropped back.
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            pad_h = (self.patch_size[0] - height % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - width % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, [0, pad_w, 0, pad_h])
            height_padded = height + pad_h
            width_padded = width + pad_w
        else:
            height_padded, width_padded = height, width

        # patchify
        x = (
            x.reshape(
                batch,
                channels,
                height_padded // self.patch_size[0],
                self.patch_size[0],
                width_padded // self.patch_size[1],
                self.patch_size[1],
            )
            .contiguous()
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .reshape(
                batch * (height_padded // self.patch_size[0]) * (width_padded // self.patch_size[1]),
                channels,
                self.patch_size[0],
                self.patch_size[1],
            )
        )
        # neural spectral: encode, transition through the latent operator, decode
        latent_token = self.latent_encoder_attn(x)
        latent_token = torch.sigmoid(latent_token) * math.pi
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        x = self.latent_decoder_attn(x, latent_token)

        # de-patchify
        x = (
            x.reshape(
                batch,
                height_padded // self.patch_size[0],
                width_padded // self.patch_size[1],
                channels,
                self.patch_size[0],
                self.patch_size[1],
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .reshape(batch, channels, height_padded, width_padded)
            .contiguous()
        )

        if height_padded != height or width_padded != width:
            x = x[:, :, :height, :width]
        return x


class LSO(nn.Module):
    """Latent Spectral Operator network for HSI-MSI fusion.

    Upstream names this class ``LSOModel``; the defaults below are upstream's
    ``DEFAULT_LSO_ARGS``.
    """

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        scale: int = 4,
        width: int = 64,
        num_token: int = 8,
        num_basis: int = 24,
        patch_size: int = 4,
        n_resblocks: int = 4,
        guide_dim: int = 128,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale}.")

        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self.width = int(width)
        self.guide_dim = int(guide_dim)
        self.n_resblocks = int(n_resblocks)
        self.patch_size = int(patch_size)

        # multiscale modules: two stride-2 rungs, so the ladder is width -> H/2 -> H/4
        self.inc = DoubleConv(self.guide_dim + 2, self.width)
        self.down1 = Down(self.width, self.width)
        self.down2 = Down(self.width, self.width)

        self.up1 = Up(self.width + self.width, self.width, self.msi_channels)
        self.up2 = Up(self.width + self.width, self.width, self.msi_channels)

        self.outc = OutConv(self.width, self.hsi_channels)

        # patchified neural spectral blocks, one per rung
        self.process1 = NeuralSpectralBlock2d(
            self.width, num_basis, (self.patch_size, self.patch_size), num_token
        )
        self.process2 = NeuralSpectralBlock2d(
            self.width, num_basis, (self.patch_size, self.patch_size), num_token
        )
        self.process3 = NeuralSpectralBlock2d(
            self.width, num_basis, (self.patch_size, self.patch_size), num_token
        )

        # the spatial encoder sees the HR-MSI and the upsampled LR-HSI stacked
        self.spatial_encoder = make_edsr_baseline(
            n_resblocks=self.n_resblocks,
            n_feats=self.guide_dim,
            n_colors=self.msi_channels + self.hsi_channels,
        )

    def get_grid(self, shape: tuple[int, int, int], device: torch.device) -> Tensor:
        """The normalized coordinate grid ``[B, H, W, 2]``."""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.view(1, size_x, 1, 1).expand(batchsize, -1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.view(1, 1, size_y, 1).expand(batchsize, size_x, -1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def _check_geometry(self, lr_hsi: Tensor, hr_msi: Tensor) -> None:
        if lr_hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f"LR HSI should carry {self.hsi_channels} bands, got {lr_hsi.shape[1]}."
            )
        if hr_msi.shape[1] != self.msi_channels:
            raise ValueError(
                f"HR MSI should carry {self.msi_channels} bands, got {hr_msi.shape[1]}."
            )
        _, _, hr_height, hr_width = hr_msi.shape
        if hr_height % 4 or hr_width % 4:
            raise ValueError(
                f"LSO needs the HR side divisible by 4 because the ladder has two "
                f"stride-2 rungs, got {hr_height}x{hr_width}."
            )
        expected_lr = (hr_height // self.scale, hr_width // self.scale)
        _, _, lr_height, lr_width = lr_hsi.shape
        if (lr_height, lr_width) != expected_lr:
            raise ValueError(
                f"LR HSI should be {expected_lr[0]}x{expected_lr[1]} for a "
                f"{hr_height}x{hr_width} HR input at scale {self.scale}, got "
                f"{lr_height}x{lr_width}."
            )

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        self._check_geometry(lr_hsi, hr_msi)

        lms = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat = torch.cat([hr_msi, lms], dim=1)
        x = self.spatial_encoder(feat)
        batch, _, height, width = x.shape
        grid = self.get_grid((batch, height, width), x.device).permute(0, 3, 1, 2)
        x = torch.cat((x, grid), dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        hr_msi2 = F.interpolate(hr_msi, scale_factor=0.5, mode="bicubic", align_corners=False)

        ns_x1 = self.process1(x1)
        ns_x2 = self.process2(x2)
        ns_x3 = self.process3(x3)

        y = self.up1(ns_x3, ns_x2, hr_msi2) + x2
        y = self.up2(y, ns_x1, hr_msi) + x1

        return lms + self.outc(y)
