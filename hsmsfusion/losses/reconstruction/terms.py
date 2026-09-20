"""Supervision and observation-consistency reconstruction losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import LOSS_REGISTRY
from hsmsfusion.losses.distance import pointwise_criterion, pointwise_distance
from hsmsfusion.losses.term import ConfigurableLossTerm


@LOSS_REGISTRY.register("l1")
class L1ReconstructionLoss(ConfigurableLossTerm):
    """Mean absolute error against the high-resolution HSI target."""

    category = "reconstruction"

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.abs(prediction - self.target(batch)))


@LOSS_REGISTRY.register("l2")
class L2ReconstructionLoss(ConfigurableLossTerm):
    """Mean squared error against the high-resolution HSI target."""

    category = "reconstruction"

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.square(prediction - self.target(batch)))


@LOSS_REGISTRY.register("char")
class CharbonnierReconstructionLoss(ConfigurableLossTerm):
    """Differentiable robust reconstruction loss."""

    category = "reconstruction"

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.epsilon = float(self.parameters.get("epsilon", 1e-3))
        if self.epsilon <= 0:
            raise ConfigurationError("char.params.epsilon must be positive.")

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        difference = prediction - self.target(batch)
        return torch.mean(torch.sqrt(torch.square(difference) + self.epsilon**2))


@LOSS_REGISTRY.register("cycle_consistency")
class CycleConsistencyLoss(ConfigurableLossTerm):
    """Re-observe a fusion prediction through the fixed benchmark operators.

    The predicted HRHS is projected to HRMS through the SRF and to LRHS through
    the blur-decimation operator, then compared to the supplied observations.
    It can therefore train dataset-level unsupervised models when the operators
    are declared as known.
    """

    category = "reconstruction"
    requires_observation_generator = True

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.spectral_weight = float(self.parameters.get("spectral_weight", 1.0))
        self.spatial_weight = float(self.parameters.get("spatial_weight", 1.0))
        self.criterion = pointwise_criterion(self.parameters, term_name="cycle_consistency")
        if self.spectral_weight < 0 or self.spatial_weight < 0:
            raise ConfigurationError("cycle_consistency weights must be non-negative.")
        if self.spectral_weight == 0 and self.spatial_weight == 0:
            raise ConfigurationError("cycle_consistency needs at least one non-zero observation weight.")

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        generated = self.require_observation_generator()(prediction)
        hr_msi = self.target(batch, "hr_msi")
        lr_hsi = self.target(batch, "lr_hsi")
        spectral = pointwise_distance(generated["hr_msi"], hr_msi, criterion=self.criterion)
        spatial = pointwise_distance(generated["lr_hsi"], lr_hsi, criterion=self.criterion)
        return self.spectral_weight * spectral + self.spatial_weight * spatial
