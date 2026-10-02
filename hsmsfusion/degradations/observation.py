"""Composition of spectral and spatial observation operators."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError, ContractError
from hsmsfusion.core.protocols import validate_nchw
from hsmsfusion.core.registry import (
    SPATIAL_OPERATOR_REGISTRY,
    SPECTRAL_OPERATOR_REGISTRY,
)

# Import concrete operator modules for their explicit registry side effects.
from hsmsfusion.degradations.spatial.gaussian_decimation import (
    GaussianDecimation as _GaussianDecimation,  # noqa: F401
)
from hsmsfusion.degradations.spectral.gaussian_srf import GaussianSRF as _GaussianSRF  # noqa: F401
from hsmsfusion.degradations.spectral.tabulated_srf import (
    TabulatedSRF as _TabulatedSRF,  # noqa: F401
)


def _plain_mapping(config: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in config.items()}


class ObservationGenerator(nn.Module):
    """Generate HR-MSI and LR-HSI from a normalized HR-HSI tensor."""

    def __init__(self, spectral_operator: nn.Module, spatial_operator: nn.Module) -> None:
        super().__init__()
        self.spectral_operator = spectral_operator
        self.spatial_operator = spatial_operator

    @property
    def scale(self) -> int:
        return int(getattr(self.spatial_operator, "scale"))

    def forward(self, hr_hsi: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_nchw(hr_hsi, name="hr_hsi")
        hr_msi = self.spectral_operator(hr_hsi)
        lr_hsi = self.spatial_operator(hr_hsi)
        expected_height = hr_hsi.shape[-2] // self.scale
        expected_width = hr_hsi.shape[-1] // self.scale
        if lr_hsi.shape[-2:] != (expected_height, expected_width):
            raise ContractError(
                "Spatial operator output does not match the configured integer scale. "
                f"Expected {(expected_height, expected_width)}, got {tuple(lr_hsi.shape[-2:])}."
            )
        return {"hr_msi": hr_msi, "lr_hsi": lr_hsi}

    @torch.no_grad()
    def generate_in_batches(
        self,
        hr_hsi: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Materialize fixed observations without retaining convolution graphs."""
        if batch_size < 1:
            raise ConfigurationError("generation_batch_size must be positive.")
        hr_msi_batches: list[torch.Tensor] = []
        lr_hsi_batches: list[torch.Tensor] = []
        for start in range(0, hr_hsi.shape[0], batch_size):
            observations = self(hr_hsi[start : start + batch_size])
            hr_msi_batches.append(observations["hr_msi"].contiguous())
            lr_hsi_batches.append(observations["lr_hsi"].contiguous())
        return {
            "hr_msi": torch.cat(hr_msi_batches, dim=0).contiguous(),
            "lr_hsi": torch.cat(lr_hsi_batches, dim=0).contiguous(),
        }


def build_observation_generator(config: Mapping[str, object]) -> ObservationGenerator:
    """Build a composable observation generator from a Hydra-compatible mapping."""
    if "spectral" not in config or "spatial" not in config:
        raise ConfigurationError("degradation config requires spectral and spatial groups.")
    spectral_config = _plain_mapping(config["spectral"])
    spatial_config = _plain_mapping(config["spatial"])
    spectral_name = str(spectral_config.pop("name"))
    spatial_name = str(spatial_config.pop("name"))
    spectral_operator = SPECTRAL_OPERATOR_REGISTRY.get(spectral_name)(spectral_config)
    spatial_operator = SPATIAL_OPERATOR_REGISTRY.get(spatial_name)(spatial_config)
    return ObservationGenerator(spectral_operator, spatial_operator)
