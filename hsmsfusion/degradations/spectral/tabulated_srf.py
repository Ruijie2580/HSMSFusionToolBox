"""Measured spectral response functions loaded from bundled MATLAB tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.protocols import validate_nchw
from hsmsfusion.core.registry import SPECTRAL_OPERATOR_REGISTRY


_BUNDLED_SRF_DIRECTORY = Path(__file__).resolve().parents[1] / "srf"


def _required(config: Mapping[str, object], name: str) -> object:
    if name not in config:
        raise ConfigurationError(f"Missing spectral operator parameter: {name}.")
    return config[name]


def _response_file(config: Mapping[str, object]) -> Path:
    requested = Path(str(_required(config, "response_file")))
    path = requested if requested.is_absolute() else _BUNDLED_SRF_DIRECTORY / requested
    if not path.is_file():
        raise ConfigurationError(f"Measured SRF file does not exist: {path}.")
    return path


def _sensor_name(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    return "".join(str(item) for item in array).strip()


def _response_table(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    try:
        contents = loadmat(path, squeeze_me=True)
    except (OSError, ValueError, NotImplementedError) as error:
        raise ConfigurationError(f"Could not read measured SRF file {path}: {error}") from error
    missing = [name for name in ("sensor", "wavelengths", "srf") if name not in contents]
    if missing:
        raise ConfigurationError(f"Measured SRF file {path} misses required variables: {missing}.")

    sensor = _sensor_name(contents["sensor"])
    wavelengths = np.asarray(contents["wavelengths"], dtype=np.float64).reshape(-1)
    response = np.asarray(contents["srf"], dtype=np.float64)
    if response.ndim != 2:
        raise ConfigurationError(f"Measured SRF array in {path} must be two-dimensional.")
    if response.shape[1] != wavelengths.size:
        if response.shape[0] == wavelengths.size:
            response = response.T
        else:
            raise ConfigurationError(
                f"Measured SRF shape {response.shape} is incompatible with {wavelengths.size} wavelengths."
            )
    if wavelengths.size < 2 or not np.all(np.isfinite(wavelengths)):
        raise ConfigurationError(f"Measured SRF wavelengths in {path} must be a finite vector of length >= 2.")
    if not np.all(np.diff(wavelengths) > 0):
        raise ConfigurationError(f"Measured SRF wavelengths in {path} must be strictly increasing.")
    if not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ConfigurationError(f"Measured SRF values in {path} must be finite and non-negative.")
    if np.any(np.sum(response, axis=1) <= 0):
        raise ConfigurationError(f"Every measured SRF band in {path} must contain positive response values.")
    return sensor, wavelengths, response


@SPECTRAL_OPERATOR_REGISTRY.register("tabulated_srf")
class TabulatedSRF(nn.Module):
    """Project HR-HSI to HR-MSI with a measured, interpolated SRF table.

    ``response_file`` is resolved against ``hsmsfusion/degradations/srf`` unless
    an absolute path is supplied. The source response is linearly interpolated
    onto the configured HSI wavelength grid and row-normalized, which preserves
    radiometric scale when applying the matrix to normalized HSI data.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        source_path = _response_file(config)
        source_sensor, source_wavelengths, source_response = _response_table(source_path)
        configured_sensor = str(config.get("sensor", source_sensor)).strip()
        if configured_sensor and configured_sensor.lower() != source_sensor.lower():
            raise ConfigurationError(
                f"Measured SRF sensor mismatch: config requests {configured_sensor!r}, "
                f"but {source_path.name} contains {source_sensor!r}."
            )

        input_channels = int(_required(config, "num_hsi_bands"))
        wavelength_start_nm = float(_required(config, "wavelength_start_nm"))
        wavelength_end_nm = float(_required(config, "wavelength_end_nm"))
        if input_channels < 1 or wavelength_end_nm <= wavelength_start_nm:
            raise ConfigurationError("Invalid HSI wavelength grid specification for tabulated_srf.")
        target_wavelengths = np.linspace(
            wavelength_start_nm,
            wavelength_end_nm,
            input_channels,
            dtype=np.float64,
        )
        interpolated = np.stack(
            [
                np.interp(
                    target_wavelengths,
                    source_wavelengths,
                    source_band,
                    left=0.0,
                    right=0.0,
                )
                for source_band in source_response
            ],
            axis=0,
        )
        row_sums = interpolated.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ConfigurationError(
                "Configured HSI wavelength grid has no overlap with one or more measured SRF bands."
            )
        interpolated /= row_sums

        expected_output_bands = config.get("num_msi_bands")
        if expected_output_bands is not None and int(expected_output_bands) != interpolated.shape[0]:
            raise ConfigurationError(
                f"Measured SRF file {source_path.name} provides {interpolated.shape[0]} MSI bands, "
                f"not num_msi_bands={expected_output_bands}."
            )

        self.sensor = source_sensor
        self.response_file = source_path.name
        self.register_buffer(
            "wavelengths_nm",
            torch.from_numpy(target_wavelengths.astype(np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "srf",
            torch.from_numpy(interpolated.astype(np.float32)),
            persistent=True,
        )

    @property
    def input_channels(self) -> int:
        return int(self.srf.shape[1])

    @property
    def output_channels(self) -> int:
        return int(self.srf.shape[0])

    def forward(self, hr_hsi: torch.Tensor) -> torch.Tensor:
        validate_nchw(hr_hsi, name="hr_hsi", channels=self.input_channels)
        return torch.einsum("mc,nchw->nmhw", self.srf.to(hr_hsi.dtype), hr_hsi)
