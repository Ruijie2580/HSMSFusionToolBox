"""Shared tensor and output contracts."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hsmsfusion.core.exceptions import ContractError


OBSERVATION_KEYS = ("lr_hsi", "hr_msi")


def observation_view(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return only observations that every fusion model may consume."""
    missing = [key for key in OBSERVATION_KEYS if key not in batch]
    if missing:
        raise ContractError(f"Batch misses required observation keys: {missing}.")
    return {key: batch[key] for key in OBSERVATION_KEYS}


def require_target(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the supervised HR-HSI target or raise a descriptive error."""
    if "gt_hsi" not in batch:
        raise ContractError("This model requires gt_hsi, but the batch does not expose it.")
    return batch["gt_hsi"]


def validate_nchw(
    tensor: torch.Tensor,
    *,
    name: str,
    channels: int | None = None,
) -> None:
    """Validate the common NCHW float tensor contract."""
    if tensor.ndim != 4:
        raise ContractError(f"{name} must have NCHW rank 4, got {tuple(tensor.shape)}.")
    if channels is not None and tensor.shape[1] != channels:
        raise ContractError(
            f"{name} must have {channels} channels, got {tensor.shape[1]}."
        )
    if not tensor.dtype.is_floating_point:
        raise ContractError(f"{name} must be floating point, got {tensor.dtype}.")


def clip_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """Apply the benchmark-wide [0, 1] prediction post-processing policy."""
    validate_nchw(prediction, name="prediction")
    if not torch.isfinite(prediction).all():
        raise ContractError("Prediction contains NaN or Inf before clipping.")
    return prediction.float().clamp(0.0, 1.0)

