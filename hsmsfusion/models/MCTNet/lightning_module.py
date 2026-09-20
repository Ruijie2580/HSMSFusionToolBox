"""Lightning wrapper for MCT-Net."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hsmsfusion.core.registry import MODEL_REGISTRY
from hsmsfusion.models.MCTNet.network import MCTNet
from hsmsfusion.models.base import BaseFusionLightningModule


@MODEL_REGISTRY.register("MCTNet")
class MCTNetLightningModule(BaseFusionLightningModule):
    """MCT-Net with standard tensor-only train, validation, test, and predict paths."""

    requires_gt = True
    supported_operator_access = {"known", "hidden"}

    def __init__(
        self,
        *,
        model_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
        metric_config: Mapping[str, Any],
        scale: int,
        operator_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            metric_config=metric_config,
            scale=scale,
            operator_config=operator_config,
        )
        self.network = MCTNet(**dict(self.model_config["params"]))

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        return self.network(lr_hsi, hr_msi)
