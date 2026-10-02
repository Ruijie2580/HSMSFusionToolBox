"""Gaussian approximation of multispectral spectral response functions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.protocols import validate_nchw
from hsmsfusion.core.registry import SPECTRAL_OPERATOR_REGISTRY


def _value(config: Mapping[str, object], name: str) -> object:
    if name not in config:
        raise ConfigurationError(f"Missing spectral operator parameter: {name}.")
    return config[name]


@SPECTRAL_OPERATOR_REGISTRY.register("gaussian_srf")
class GaussianSRF(nn.Module):
    """Convert HR-HSI to HR-MSI with row-normalized Gaussian SRFs."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        centers = torch.as_tensor(_value(config, "centers_nm"), dtype=torch.float32)
        fwhm = torch.as_tensor(_value(config, "fwhm_nm"), dtype=torch.float32)
        if centers.ndim != 1 or fwhm.ndim != 1 or centers.numel() != fwhm.numel():
            raise ConfigurationError("centers_nm and fwhm_nm must be equal-length vectors.")
        if torch.any(fwhm <= 0):
            raise ConfigurationError("All SRF FWHM values must be positive.")

        bands = int(_value(config, "num_hsi_bands"))
        start_nm = float(_value(config, "wavelength_start_nm"))
        end_nm = float(_value(config, "wavelength_end_nm"))
        if bands < 1 or end_nm <= start_nm:
            raise ConfigurationError("Invalid HSI wavelength grid specification.")

        wavelengths = torch.linspace(start_nm, end_nm, bands, dtype=torch.float32)
        sigma = fwhm / 2.354820045
        response = torch.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / sigma[:, None]) ** 2)
        response = response / response.sum(dim=1, keepdim=True).clamp_min(1e-12)

        self.register_buffer("wavelengths_nm", wavelengths, persistent=True)
        self.register_buffer("srf", response, persistent=True)

    @property
    def input_channels(self) -> int:
        return int(self.srf.shape[1])

    @property
    def output_channels(self) -> int:
        return int(self.srf.shape[0])

    def forward(self, hr_hsi: torch.Tensor) -> torch.Tensor:
        validate_nchw(hr_hsi, name="hr_hsi", channels=self.input_channels)
        return torch.einsum("mc,nchw->nmhw", self.srf.to(hr_hsi.dtype), hr_hsi)

