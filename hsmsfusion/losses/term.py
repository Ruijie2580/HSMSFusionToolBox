"""Shared contracts for configurable loss terms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.degradations.observation import ObservationGenerator


class ConfigurableLossTerm(nn.Module):
    """Base class for one registry-backed loss term.

    Subclasses belong to either the ``reconstruction`` or ``regularization``
    configuration group and receive only their own ``params`` mapping.
    """

    category: str
    requires_observation_generator = False

    def __init__(
        self,
        parameters: Mapping[str, Any] | None = None,
        observation_generator: ObservationGenerator | None = None,
    ) -> None:
        super().__init__()
        self.parameters = dict(parameters or {})
        self.observation_generator = observation_generator

    def forward(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def target(batch: Mapping[str, torch.Tensor], key: str = "gt_hsi") -> torch.Tensor:
        target = batch.get(key)
        if not isinstance(target, torch.Tensor):
            raise ConfigurationError(
                f"Loss term requires batch['{key}'] as a tensor, but it was not provided."
            )
        return target

    def require_observation_generator(self) -> ObservationGenerator:
        if self.observation_generator is None:
            raise ConfigurationError(
                "This loss term requires known spectral and spatial operators. "
                "Set degradation.operator_access=known and configure both operators."
            )
        return self.observation_generator
