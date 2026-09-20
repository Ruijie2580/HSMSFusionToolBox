"""Reusable pointwise distances for configurable reconstruction constraints."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from hsmsfusion.core.exceptions import ConfigurationError


def pointwise_criterion(parameters: dict[str, object], *, term_name: str) -> str:
    """Read a shared ``criterion: l1 | mse`` option from a loss parameter map."""

    criterion = str(parameters.get("criterion", "l1")).lower()
    if criterion not in {"l1", "mse"}:
        raise ConfigurationError(f"{term_name}.params.criterion must be either 'l1' or 'mse'.")
    return criterion


def pointwise_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion: str,
) -> torch.Tensor:
    """Compute a mean L1 or mean-squared distance without duplicated branching."""

    if criterion == "l1":
        return functional.l1_loss(prediction, target)
    if criterion == "mse":
        return functional.mse_loss(prediction, target)
    raise ConfigurationError(f"Unsupported pointwise criterion {criterion!r}.")
