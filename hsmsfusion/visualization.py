"""Training-time visual artifacts for the fixed fusion observation protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

_RGB_WAVELENGTHS_NM = (650.0, 550.0, 450.0)


def save_srf_curve(spectral_operator: Any, output_path: Path) -> bool:
    """Write the actual fixed spectral response functions used by a run.

    Returns ``False`` for operators that do not expose a wavelength grid and a
    response matrix, allowing third-party spectral operators to opt out safely.
    """

    wavelengths = getattr(spectral_operator, "wavelengths_nm", None)
    responses = getattr(spectral_operator, "srf", None)
    if not isinstance(wavelengths, torch.Tensor) or not isinstance(responses, torch.Tensor):
        return False
    if wavelengths.ndim != 1 or responses.ndim != 2 or responses.shape[1] != wavelengths.numel():
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavelength_values = wavelengths.detach().float().cpu().numpy()
    response_values = responses.detach().float().cpu().numpy()
    figure, axis = plt.subplots(figsize=(10.0, 4.6), constrained_layout=False)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.72)
    for index, response in enumerate(response_values):
        axis.plot(
            wavelength_values,
            response,
            linewidth=1.7,
            label=f"MSI band {index + 1}",
        )
    figure.suptitle("Spectral response functions", fontsize=13, fontweight="bold", y=0.83)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Normalized response")
    axis.set_xlim(float(wavelength_values.min()), float(wavelength_values.max()))
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25, linewidth=0.7)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=min(4, response_values.shape[0]),
        fontsize=8,
        frameon=False,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return True


def save_validation_comparison(
    batch: Mapping[str, torch.Tensor],
    prediction: torch.Tensor,
    output_path: Path,
    *,
    spectral_operator: Any | None = None,
) -> None:
    """Save all samples from the first validation batch in compact four-panel groups."""

    target = _nchw_batch(batch["gt_hsi"])
    lr_hsi = _nchw_batch(batch["lr_hsi"])
    hr_msi = _nchw_batch(batch["hr_msi"])
    fused = _nchw_batch(prediction)
    sample_count = target.shape[0]
    if (
        lr_hsi.shape[0] != sample_count
        or hr_msi.shape[0] != sample_count
        or fused.shape[0] != sample_count
    ):
        raise ValueError("Validation visualization tensors must share the same batch size.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hsi_wavelengths = _operator_tensor(spectral_operator, "wavelengths_nm")
    srf = _operator_tensor(spectral_operator, "srf")
    hsi_indices = _rgb_band_indices(
        hsi_wavelengths,
        target.shape[1],
    )
    msi_indices = _msi_rgb_band_indices(
        hsi_wavelengths,
        srf,
        hr_msi.shape[1],
    )

    lr_for_display = functional.interpolate(
        lr_hsi,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    images = (
        ("HRHS", _to_rgb_batch(target, hsi_indices)),
        ("LRHS (upsampled)", _to_rgb_batch(lr_for_display, hsi_indices)),
        ("HRMS", _to_rgb_batch(hr_msi, msi_indices)),
        ("Fused HSI", _to_rgb_batch(fused, hsi_indices)),
    )

    groups_per_row = 2
    panels_per_sample = len(images)
    row_count = int(np.ceil(sample_count / groups_per_row))
    figure = plt.figure(
        figsize=(16.0, max(2.4, 2.0 * row_count)),
        constrained_layout=False,
    )
    grid = figure.add_gridspec(
        row_count,
        groups_per_row * panels_per_sample,
        left=0.008,
        right=0.992,
        bottom=0.006,
        top=0.955,
        wspace=0.025,
        hspace=0.08,
    )
    for sample_index in range(sample_count):
        row_index, group_index = divmod(sample_index, groups_per_row)
        column_offset = group_index * panels_per_sample
        for panel_index, (title, image_batch) in enumerate(images):
            axis = figure.add_subplot(grid[row_index, column_offset + panel_index])
            axis.imshow(image_batch[sample_index])
            if row_index == 0:
                axis.set_title(title, fontsize=9, fontweight="bold", pad=3)
            if panel_index == 0:
                axis.text(
                    0.02,
                    0.98,
                    f"#{sample_index:02d}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.4, "edgecolor": "none"},
                )
            axis.set_axis_off()
    figure.suptitle(
        f"Validation first batch ({sample_count} samples)",
        fontsize=11,
        fontweight="bold",
        y=0.994,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def _nchw_batch(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] < 1:
        raise ValueError("Validation visualization expects non-empty NCHW tensors.")
    return tensor.detach().float().cpu()


def _operator_tensor(operator: Any | None, name: str) -> torch.Tensor | None:
    value = getattr(operator, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _rgb_band_indices(
    wavelengths_nm: torch.Tensor | None,
    channel_count: int,
) -> list[int]:
    if (
        isinstance(wavelengths_nm, torch.Tensor)
        and wavelengths_nm.ndim == 1
        and wavelengths_nm.numel() == channel_count
    ):
        wavelengths = wavelengths_nm.detach().float().cpu()
        return [
            int(torch.argmin(torch.abs(wavelengths - target_nm)).item())
            for target_nm in _RGB_WAVELENGTHS_NM
        ]
    return _fallback_rgb_indices(channel_count)


def _msi_rgb_band_indices(
    wavelengths_nm: torch.Tensor | None,
    srf: torch.Tensor | None,
    channel_count: int,
) -> list[int]:
    if (
        isinstance(wavelengths_nm, torch.Tensor)
        and isinstance(srf, torch.Tensor)
        and wavelengths_nm.ndim == 1
        and srf.ndim == 2
        and srf.shape == (channel_count, wavelengths_nm.numel())
    ):
        centers = (srf.detach().float().cpu() * wavelengths_nm.detach().float().cpu()).sum(dim=1)
        return [
            int(torch.argmin(torch.abs(centers - target_nm)).item())
            for target_nm in _RGB_WAVELENGTHS_NM
        ]
    return _fallback_rgb_indices(channel_count)


def _fallback_rgb_indices(channel_count: int) -> list[int]:
    if channel_count < 1:
        raise ValueError("Cannot render an image with zero channels.")
    return np.linspace(0, channel_count - 1, num=3).round().astype(int).tolist()


def _to_rgb_batch(cube: torch.Tensor, channel_indices: list[int]) -> np.ndarray:
    if cube.ndim != 4:
        raise ValueError("Visualization expects an NCHW image tensor.")
    rgb = cube[:, channel_indices].permute(0, 2, 3, 1).contiguous().numpy()
    return np.clip(rgb, 0.0, 1.0)
