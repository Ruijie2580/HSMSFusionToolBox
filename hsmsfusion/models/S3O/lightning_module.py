"""Lightning training wrapper for S3O."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hsmsfusion.core.registry import MODEL_REGISTRY
from hsmsfusion.models.base import BaseFusionLightningModule
from hsmsfusion.models.S3O.network import S3O


@MODEL_REGISTRY.register("S3O")
class S3OLightningModule(BaseFusionLightningModule):
    """Selective Spatial-Spectral Operator for HSI-MSI fusion.

    Beyond the network, this carries upstream's progressive-enhancement
    curriculum: the fraction of tokens routed through the ``enhance`` head is
    held at zero for the first ``warmup_fraction`` of training, ramped linearly
    to ``max_ratio`` over the next ``ramp_fraction``, and held there.  Upstream
    applies it from the training script with ``--sel_warmup 0.10 --sel_ramp 0.20
    --sel_max 0.10``; here it is read from ``model.progressive_enhancement`` so
    it travels with the recipe, and the same values are the defaults.
    """

    requires_gt = True
    supported_operator_access = {"known", "hidden"}

    #: Upstream's ``--sel_warmup`` / ``--sel_ramp`` / ``--sel_max``.
    DEFAULT_ENHANCEMENT: Mapping[str, float] = {
        "warmup_fraction": 0.10,
        "ramp_fraction": 0.20,
        "max_ratio": 0.10,
    }

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
        enhancement = dict(self.DEFAULT_ENHANCEMENT)
        enhancement.update(dict(self.model_config.get("progressive_enhancement") or {}))
        self.sel_warmup_fraction = float(enhancement["warmup_fraction"])
        self.sel_ramp_fraction = float(enhancement["ramp_fraction"])
        self.sel_max_ratio = float(enhancement["max_ratio"])

        parameters = dict(self.model_config["params"])
        # Blocks start at the plateau, the value upstream's test-time model uses.
        # The curriculum overwrites this at the start of every training epoch, so
        # a run that evaluates a checkpoint without training keeps the plateau.
        parameters["sel_ratio"] = self.sel_max_ratio
        self.network = S3O(**parameters)

    def enhancement_ratio_for_epoch(self, epoch: int, total_epochs: int) -> float:
        """Return the selective-enhancement ratio for ``epoch``.

        Upstream computes the two boundaries once from the total epoch count, so
        they do not move if training is cut short -- this reproduces that.
        """
        warmup_epochs = int(total_epochs * self.sel_warmup_fraction)
        ramp_epochs = int(total_epochs * self.sel_ramp_fraction)
        if epoch < warmup_epochs:
            return 0.0
        if epoch < warmup_epochs + ramp_epochs:
            progress = (epoch - warmup_epochs) / ramp_epochs
            return self.sel_max_ratio * progress
        return self.sel_max_ratio

    def on_train_epoch_start(self) -> None:
        ratio = self.enhancement_ratio_for_epoch(self.current_epoch, self.trainer.max_epochs)
        self.network.set_enhancement_ratio(ratio)
        self.log("train/sel_ratio", ratio, on_step=False, on_epoch=True)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        return self.network(lr_hsi, hr_msi)
