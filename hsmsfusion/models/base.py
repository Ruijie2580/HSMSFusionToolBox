"""Base LightningModule protocol for supervised and unsupervised fusion models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningModule

from hsmsfusion.core.exceptions import ConfigurationError, ContractError
from hsmsfusion.core.protocols import clip_prediction, observation_view, require_target
from hsmsfusion.losses import LossManager
from hsmsfusion.metrics.evaluator import Evaluator


class BaseFusionLightningModule(LightningModule):
    """Model-owned optimization hooks with a shared validation metric protocol."""

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
        super().__init__()
        self.model_config = dict(model_config)
        self.optimizer_config = dict(optimizer_config)
        self.scheduler_config = dict(scheduler_config)
        self.loss_config = dict(loss_config)
        self.metric_config = dict(metric_config)
        self.scale = int(scale)
        self.known_operator_config = (
            dict(operator_config) if operator_config is not None else None
        )
        self.loss_manager = LossManager(
            self.loss_config,
            operator_config=self.known_operator_config,
        )
        self._validation_predictions: list[torch.Tensor] = []
        self._validation_targets: list[torch.Tensor] = []
        self.save_hyperparameters(
            {
                "model_name": self.model_config.get("name"),
                "scale": self.scale,
                "requires_gt": self.requires_gt,
            }
        )
        inference_config = dict(self.model_config.get("inference", {}))
        self.inference_mode = str(inference_config.get("mode", "tiled"))
        self.tile_size = int(inference_config.get("tile_size", 0))
        self.shave = int(inference_config.get("shave", 0))
        if self.inference_mode not in ("tiled", "full"):
            raise ConfigurationError(
                f"inference.mode must be 'tiled' or 'full', got {self.inference_mode!r}"
            )
        if self.inference_mode == "tiled" and self.tile_size > 0:
            pass  # explicit tiled mode with tile_size — OK
        elif self.inference_mode == "tiled":
            # 未指定 tile_size 时退化为全图推理（向后兼容旧 run）
            self.inference_mode = "full"

    def require_known_operator_config(self) -> Mapping[str, Any]:
        """Expose the fixed observation protocol only in known-operator runs."""
        if self.known_operator_config is None:
            raise ContractError(
                "This run uses hidden operators; the true SRF/PSF configuration is unavailable."
            )
        return self.known_operator_config

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_objective(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compose configured standard loss terms for supervised or unsupervised runs.

        A model may override this only when its objective needs non-standard
        intermediate tensors that cannot be expressed as registered loss terms.
        """

        return self.loss_manager(prediction, batch)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        observations = observation_view(batch)
        prediction = self(**observations)
        objective_batch = dict(observations)
        if self.requires_gt:
            objective_batch["gt_hsi"] = require_target(batch)
        total_loss, components = self.compute_objective(prediction, objective_batch)
        if total_loss.ndim != 0 or not torch.isfinite(total_loss):
            raise ContractError("compute_objective must return a finite scalar total loss.")
        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in components.items():
            self.log(f"train/{name}", value, on_step=False, on_epoch=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions.clear()
        self._validation_targets.clear()

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> Mapping[str, torch.Tensor]:
        observations = observation_view(batch)
        raw_prediction = self(**observations)
        objective_batch = dict(observations)
        if self.requires_gt:
            objective_batch["gt_hsi"] = require_target(batch)
        total_loss, components = self.compute_objective(raw_prediction, objective_batch)
        if total_loss.ndim != 0 or not torch.isfinite(total_loss):
            raise ContractError("compute_objective must return a finite scalar validation loss.")
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False)
        for name, value in components.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True)
        prediction = clip_prediction(raw_prediction)
        self._validation_predictions.append(prediction.detach().cpu())
        self._validation_targets.append(require_target(batch).detach().float().cpu())
        return {"prediction": prediction.detach()}

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        prediction = torch.cat(self._validation_predictions).numpy().astype(np.float32, copy=False)
        target = torch.cat(self._validation_targets).numpy().astype(np.float32, copy=False)
        result = Evaluator(self.metric_config, scale=self.scale).evaluate(prediction, target)
        for metric_name, item in result.summary["metrics"].items():
            self.log(
                f"val/{metric_name}",
                float(item["mean"]),
                on_step=False,
                on_epoch=True,
                prog_bar=metric_name == "psnr_global",
            )

    def predict_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        inputs = observation_view(batch)
        if self.inference_mode == "tiled" and self.tile_size > 0:
            return self._tiled_predict(inputs)
        return clip_prediction(self(**inputs)).detach().cpu()

    def _tiled_predict(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Split image into overlapping tiles, predict each, stitch valid centers."""
        lr_hsi = inputs["lr_hsi"]
        hr_msi = inputs["hr_msi"]
        B, _, H, W = hr_msi.shape
        scale = self.scale
        tile = self.tile_size
        shave = self.shave

        if tile >= H and tile >= W:
            return clip_prediction(self(**inputs)).detach().cpu()

        step = tile - 2 * shave
        output = None
        count = hr_msi.new_zeros(B, 1, H, W)

        for y in range(0, H, step):
            for x in range(0, W, step):
                y_start = y
                y_end = min(y + tile, H)
                x_start = x
                x_end = min(x + tile, W)
                if y_end - y_start < tile:
                    y_start = max(y_end - tile, 0)
                if x_end - x_start < tile:
                    x_start = max(x_end - tile, 0)

                hr_tile = hr_msi[:, :, y_start:y_end, x_start:x_end]
                lr_y_start = y_start // scale
                lr_y_end = y_end // scale
                lr_x_start = x_start // scale
                lr_x_end = x_end // scale
                lr_tile = lr_hsi[:, :, lr_y_start:lr_y_end, lr_x_start:lr_x_end]

                pred_tile = clip_prediction(
                    self(**{"lr_hsi": lr_tile, "hr_msi": hr_tile})
                )

                if output is None:
                    output = pred_tile.new_zeros(B, pred_tile.shape[1], H, W)

                vy_start = shave if y > 0 else 0
                vy_end = pred_tile.shape[2] - (shave if y + tile < H else 0)
                vx_start = shave if x > 0 else 0
                vx_end = pred_tile.shape[3] - (shave if x + tile < W else 0)

                oy_start = y_start + vy_start
                oy_end = y_start + vy_end
                ox_start = x_start + vx_start
                ox_end = x_start + vx_end

                output[:, :, oy_start:oy_end, ox_start:ox_end] += \
                    pred_tile[:, :, vy_start:vy_end, vx_start:vx_end]
                count[:, :, oy_start:oy_end, ox_start:ox_end] += 1.0

        return (output / count.clamp_min(1.0)).detach().cpu()

    def configure_optimizers(self) -> Any:
        optimizer_name = str(self.optimizer_config["name"]).lower()
        learning_rate = float(self.optimizer_config["lr"])
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.0))
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise ConfigurationError("Selected model has no trainable parameters.")
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=tuple(self.optimizer_config.get("betas", (0.9, 0.999))),
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=tuple(self.optimizer_config.get("betas", (0.9, 0.999))),
                weight_decay=weight_decay,
            )
        else:
            raise ConfigurationError(f"Unsupported optimizer: {optimizer_name!r}.")

        scheduler_name = str(self.scheduler_config.get("name", "none")).lower()
        if scheduler_name == "none":
            return optimizer
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_config["t_max"]),
                eta_min=float(self.scheduler_config.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        if scheduler_name in {"multistep", "multisteplr", "multi_step", "multi_step_lr"}:
            raw_milestones = self.scheduler_config.get("milestones")
            if raw_milestones is None:
                raise ConfigurationError(
                    "MultiStepLR requires a non-empty 'milestones' sequence."
                )
            try:
                milestones = [int(milestone) for milestone in raw_milestones]
            except (TypeError, ValueError) as error:
                raise ConfigurationError(
                    "MultiStepLR 'milestones' must be a sequence of integers."
                ) from error
            if not milestones or any(milestone < 1 for milestone in milestones):
                raise ConfigurationError(
                    "MultiStepLR 'milestones' must contain positive integers."
                )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=float(self.scheduler_config.get("gamma", 0.1)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        raise ConfigurationError(f"Unsupported scheduler: {scheduler_name!r}.")
