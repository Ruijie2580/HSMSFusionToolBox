"""SCSRNet adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/zsspo/SCSRNet
Paper: SCSRNet: A multi-scale spatial-conditioned spectral routing network for
hyperspectral and multispectral image fusion, International Journal of Applied
Earth Observation and Geoinformation 153 (2026), 105547.
DOI: https://doi.org/10.1016/j.jag.2026.105547

Three upstream designs are preserved here:
  - DFG     : DynamicFusionGate, a sigmoid gate that balances the HSI and MSI
              contributions before the residual trunk.
  - LK-SSRB : LKSSBlock with a large-kernel MSPM spatial branch and an SCSRM
              (SpatialGuidedSpectralRouting) spectral branch.
  - RSC     : a per-channel residual scaling vector initialised to 1e-6.

HSMS adaptation notes
---------------------
The original implementation reads its channel counts from a training config
dictionary and targets a three-band RGB MSI.  This port turns the channel
counts, scale, and block topology into explicit constructor arguments and
follows the toolbox protocol (Chikusei: 128 HSI / 8 WV2 MSI bands), so
``msi_layer`` consumes the full 8-band MSI instead of 3 RGB channels.

Removed relative to upstream, with forward numerics unchanged:
  - the optional ``depthwise_conv2d_implicit_gemm`` fast path and every debug
    ``print``,
  - the deploy-time branch folding (``fuse_bn``, ``merge_dilated_branches``,
    ``merge_dilated_into_large_kernel``, ``reparameterize``) -- it is a
    weight-folding optimisation that is mathematically equivalent to the
    multi-branch form kept here, and upstream ``LKSSBlock.reparameterize``
    even referenced a ``self.dwconv`` attribute that the class never defines,
  - the OpenMMLab-style ``init_weights`` hooks and the ``with_cp`` activation
    checkpoint switch, neither of which upstream ever calls.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


class DropPath(nn.Module):
    """Per-sample stochastic depth; an identity when ``drop_prob`` is zero."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def _depthwise_conv2d(
    channels: int,
    kernel_size: int | Sequence[int],
    stride: int = 1,
    padding: int | Sequence[int] | None = None,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """Depthwise 2D convolution; the upstream iGEMM fast path is not ported."""
    kernel_size = _to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = _to_2tuple(padding)
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=bias,
    )


class MSPM(nn.Module):
    """Multi-scale spatial perception module (the LK-SSRB spatial branch).

    A large depthwise convolution plus a set of dilated depthwise branches
    summed into it.  Upstream folds every branch into one equivalent large
    kernel before deployment; that path is not ported, so the multi-branch
    training-time form is always used.
    """

    _BRANCH_TABLE: dict[int, tuple[list[int], list[int]]] = {
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        9: ([5, 5, 3, 3], [1, 2, 3, 4]),
        7: ([5, 3, 3], [1, 2, 3]),
        5: ([3, 3], [1, 2]),
    }

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size not in self._BRANCH_TABLE:
            raise ValueError(f"Undefined kernel sizes for kernel_size={kernel_size}.")
        self.kernel_sizes, self.dilates = self._BRANCH_TABLE[kernel_size]
        self.lk_origin = _depthwise_conv2d(channels, kernel_size, padding=kernel_size // 2)
        self.origin_bn = nn.BatchNorm2d(channels)
        for kernel, dilation in zip(self.kernel_sizes, self.dilates):
            self.add_module(
                f"dil_conv_k{kernel}_{dilation}",
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=(dilation * (kernel - 1) + 1) // 2,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                ),
            )
            self.add_module(f"dil_bn_k{kernel}_{dilation}", nn.BatchNorm2d(channels))

    def forward(self, x: Tensor) -> Tensor:
        out = self.origin_bn(self.lk_origin(x))
        for kernel, dilation in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{kernel}_{dilation}")
            norm = getattr(self, f"dil_bn_k{kernel}_{dilation}")
            out = out + norm(conv(x))
        return out


class DynamicFusionGate(nn.Module):
    """DFG: a learned sigmoid gate over concatenated HSI and MSI features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x_hsi: Tensor, x_msi: Tensor) -> Tensor:
        weight = self.gate(torch.cat([x_hsi, x_msi], dim=1))
        fused = weight * x_hsi + (1 - weight) * x_msi
        return self.refine(fused)


class ChannelConvBranch(nn.Module):
    """1D convolution along the channel dimension, independent per pixel."""

    def __init__(self, channels: int, kernel_size: int, activation: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            bias=False,
        )
        self.act = nn.GELU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        out = self.conv(x.reshape(batch, channels, height * width).unsqueeze(1))
        return self.act(out).squeeze(1).reshape(batch, channels, height, width)


class SpatialGuidedSpectralRouting(nn.Module):
    """SCSRM: softmax-routed multi-scale spectral (channel-wise) convolution."""

    def __init__(
        self,
        channels: int,
        branch_configs: Sequence[tuple[int, bool]] = ((3, True), (5, True), (7, True)),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.routing = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, len(branch_configs), 1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.spectral_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                ChannelConvBranch(channels=channels, kernel_size=kernel, activation=activation)
                for kernel, activation in branch_configs
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        route = self.softmax(self.routing(x))
        projected = self.spectral_proj(x)
        out = projected
        for index, branch in enumerate(self.branches):
            out = out + route[:, index : index + 1, :, :] * branch(projected)
        return out


class LKSSBlock(nn.Module):
    """Large-kernel spatial-spectral residual block (LK-SSRB).

    ``kernel_size >= 7`` builds the large-kernel MSPM spatial branch, while
    ``kernel_size`` of 3 or 5 builds a plain depthwise convolution (SK-SSRB).
    The residual is rescaled per channel by ``gamma`` (the RSC connection).
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        drop_path: float = 0.0,
        scale_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size == 0:
            self.spa = nn.Identity()
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.spa = MSPM(dim, kernel_size)
            self.norm = nn.BatchNorm2d(dim)
        elif kernel_size in (3, 5):
            self.spa = _depthwise_conv2d(dim, kernel_size, padding=kernel_size // 2)
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}.")

        self.spec = SpatialGuidedSpectralRouting(dim)
        self.gamma = (
            nn.Parameter(scale_value * torch.ones(dim), requires_grad=True)
            if scale_value is not None and scale_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm(self.spa(x))
        residual = self.spec(residual)
        if self.gamma is not None:
            residual = self.gamma.reshape(1, -1, 1, 1) * residual
        return x + self.drop_path(residual)


_DEFAULT_KERNEL_SIZES_BY_DEPTHS: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {
    (2, 2): ((3, 3), (13, 13)),
}


class SCSRNet(nn.Module):
    """SCSRNet for hyperspectral and multispectral image fusion.

    The trunk predicts a residual over the bicubic-upsampled LR HSI, so the
    final output is ``out(x) + hsi_up``.
    """

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        scale: int = 4,
        dims: Sequence[int] = (64, 64),
        depths: Sequence[int] = (2, 2),
        kernel_sizes: Sequence[Sequence[int]] | None = None,
        drop_path_rate: float = 0.0,
        scale_value: float = 1e-6,
    ) -> None:
        super().__init__()
        dims = tuple(int(dim) for dim in dims)
        depths = tuple(int(depth) for depth in depths)
        if len(dims) != len(depths):
            raise ValueError("dims and depths must have the same length.")

        if kernel_sizes is None:
            if depths not in _DEFAULT_KERNEL_SIZES_BY_DEPTHS:
                raise ValueError(f"No default kernel sizes for depths {depths}.")
            kernel_sizes = _DEFAULT_KERNEL_SIZES_BY_DEPTHS[depths]
        kernel_sizes = tuple(tuple(int(kernel) for kernel in stage) for stage in kernel_sizes)
        for stage, depth in zip(kernel_sizes, depths):
            if len(stage) != depth:
                raise ValueError("kernel sizes do not match the depths.")

        self.hsi_channels = int(hsi_channels)
        self.out_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)

        drop_path_rates = torch.linspace(0.0, float(drop_path_rate), sum(depths)).tolist()

        first_channels = dims[0]
        last_channels = dims[-1]
        self.hsi_layer = nn.Conv2d(self.hsi_channels, first_channels, kernel_size=3, padding=1)
        self.msi_layer = nn.Conv2d(self.msi_channels, first_channels, kernel_size=3, padding=1)
        self.gate = DynamicFusionGate(first_channels)

        blocks = []
        cursor = 0
        for dim, depth, stage in zip(dims, depths, kernel_sizes):
            blocks.append(
                nn.Sequential(
                    *[
                        LKSSBlock(
                            dim=dim,
                            kernel_size=stage[index],
                            drop_path=drop_path_rates[cursor + index],
                            scale_value=scale_value,
                        )
                        for index in range(depth)
                    ]
                )
            )
            cursor += depth
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(last_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        hsi_up = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic")
        x_hsi = self.hsi_layer(hsi_up)
        x_msi = self.msi_layer(hr_msi)
        x = self.gate(x_hsi, x_msi)
        for block in self.blocks:
            x = block(x)
        return self.out(x) + hsi_up
