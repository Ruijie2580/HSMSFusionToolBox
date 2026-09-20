"""Gaussian blur followed by fixed-phase decimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import SPATIAL_OPERATOR_REGISTRY
from hsmsfusion.core.protocols import validate_nchw


@SPATIAL_OPERATOR_REGISTRY.register("gaussian_decimation")
class GaussianDecimation(nn.Module):
    """Apply a shared isotropic Gaussian PSF and decimate at offset [0, 0]."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.scale = int(config["scale"])
        self.kernel_size = int(config["kernel_size"])
        self.sigma = float(config["sigma"])
        self.padding = str(config.get("padding", "reflect"))
        offset = tuple(int(value) for value in config.get("offset", (0, 0)))
        if self.scale < 1:
            raise ConfigurationError("Spatial scale must be at least one.")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ConfigurationError("kernel_size must be a positive odd integer.")
        if self.sigma <= 0:
            raise ConfigurationError("Gaussian sigma must be positive.")
        if offset != (0, 0):
            raise ConfigurationError("The benchmark requires spatial offset [0, 0].")
        if self.padding not in {"reflect", "replicate", "circular", "zeros"}:
            raise ConfigurationError(f"Unsupported padding mode: {self.padding!r}.")

        radius = (self.kernel_size - 1) / 2
        coordinate = torch.arange(self.kernel_size, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-0.5 * (coordinate / self.sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d[None, None], persistent=True)

    def forward(self, hr_hsi: torch.Tensor) -> torch.Tensor:
        validate_nchw(hr_hsi, name="hr_hsi")
        channels = hr_hsi.shape[1]
        weight = self.kernel.to(dtype=hr_hsi.dtype).expand(channels, 1, -1, -1)
        padding = self.kernel_size // 2
        if self.padding == "zeros":
            blurred = functional.conv2d(hr_hsi, weight, padding=padding, groups=channels)
        else:
            padded = functional.pad(
                hr_hsi,
                (padding, padding, padding, padding),
                mode=self.padding,
            )
            blurred = functional.conv2d(padded, weight, groups=channels)
        return blurred[:, :, 0 :: self.scale, 0 :: self.scale]

