"""Spectral and spatial regularization losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as functional

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import LOSS_REGISTRY
from hsmsfusion.losses.distance import pointwise_criterion, pointwise_distance
from hsmsfusion.losses.term import ConfigurableLossTerm


@LOSS_REGISTRY.register("sam")
class SAMRegularizationLoss(ConfigurableLossTerm):
    """Mean spectral-angle loss in radians against the HRHS reference."""

    category = "regularization"

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.epsilon = float(self.parameters.get("epsilon", 1e-8))
        if self.epsilon <= 0:
            raise ConfigurationError("sam.params.epsilon must be positive.")

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        target = self.target(batch)
        dot = torch.sum(prediction * target, dim=1)
        pred_norm = torch.linalg.vector_norm(prediction, dim=1)
        target_norm = torch.linalg.vector_norm(target, dim=1)
        cosine = dot / (pred_norm * target_norm + self.epsilon)
        return torch.mean(torch.acos(torch.clamp(cosine, -1.0 + self.epsilon, 1.0 - self.epsilon)))


@LOSS_REGISTRY.register("ssim")
class SSIMRegularizationLoss(ConfigurableLossTerm):
    """Differentiable per-band two-dimensional SSIM loss."""

    category = "regularization"

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.window_size = int(self.parameters.get("window_size", 11))
        self.sigma = float(self.parameters.get("sigma", 1.5))
        self.k1 = float(self.parameters.get("k1", 0.01))
        self.k2 = float(self.parameters.get("k2", 0.03))
        self.epsilon = float(self.parameters.get("epsilon", 1e-8))
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ConfigurationError("ssim.params.window_size must be a positive odd integer.")
        if self.sigma <= 0 or self.k1 <= 0 or self.k2 <= 0 or self.epsilon <= 0:
            raise ConfigurationError("ssim parameters sigma, k1, k2, and epsilon must be positive.")

    def _window(self, channels: int, prediction: torch.Tensor) -> torch.Tensor:
        coordinate = torch.arange(
            self.window_size,
            device=prediction.device,
            dtype=prediction.dtype,
        ) - (self.window_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * torch.square(coordinate / self.sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.expand(channels, 1, self.window_size, self.window_size)

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        target = self.target(batch)
        if prediction.ndim != 4:
            raise ConfigurationError("ssim expects NCHW prediction tensors.")
        height, width = prediction.shape[-2:]
        padding = self.window_size // 2
        if height <= padding or width <= padding:
            raise ConfigurationError("ssim window_size is too large for the prediction spatial size.")

        channels = prediction.shape[1]
        window = self._window(channels, prediction)

        def convolve(value: torch.Tensor) -> torch.Tensor:
            padded = functional.pad(value, (padding, padding, padding, padding), mode="reflect")
            return functional.conv2d(padded, window, groups=channels)

        mean_prediction = convolve(prediction)
        mean_target = convolve(target)
        variance_prediction = convolve(prediction * prediction) - mean_prediction.square()
        variance_target = convolve(target * target) - mean_target.square()
        covariance = convolve(prediction * target) - mean_prediction * mean_target
        c1 = self.k1**2
        c2 = self.k2**2
        numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
        denominator = (mean_prediction.square() + mean_target.square() + c1) * (
            variance_prediction + variance_target + c2
        )
        ssim = numerator / denominator.clamp_min(self.epsilon)
        return 1.0 - torch.mean(ssim)


@LOSS_REGISTRY.register("tv")
class TotalVariationRegularizationLoss(ConfigurableLossTerm):
    """Anisotropic spatial total-variation penalty without a reference target.

    ``params.criterion`` selects an L1 or MSE penalty on the spatial gradients.
    """

    category = "regularization"

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.criterion = pointwise_criterion(self.parameters, term_name="tv")

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        del batch
        if prediction.ndim != 4:
            raise ConfigurationError("tv expects NCHW prediction tensors.")
        if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
            raise ConfigurationError("tv requires prediction height and width of at least 2.")
        vertical_difference = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        horizontal_difference = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        vertical = pointwise_distance(
            vertical_difference,
            torch.zeros_like(vertical_difference),
            criterion=self.criterion,
        )
        horizontal = pointwise_distance(
            horizontal_difference,
            torch.zeros_like(horizontal_difference),
            criterion=self.criterion,
        )
        return vertical + horizontal


class _TargetGradientRegularizationLoss(ConfigurableLossTerm):
    """Shared target-gradient loss with a configurable pointwise criterion."""

    category = "regularization"

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(parameters=parameters, **kwargs)
        self.criterion = pointwise_criterion(self.parameters, term_name=self.registry_name)


@LOSS_REGISTRY.register("spatial_edge")
class SpatialEdgeRegularizationLoss(_TargetGradientRegularizationLoss):
    """Match vertical and horizontal spatial gradients of prediction and HRHS."""

    registry_name = "spatial_edge"

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        target = self.target(batch)
        if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
            raise ConfigurationError("spatial_edge requires prediction height and width of at least 2.")
        vertical_prediction = prediction[:, :, :-1, :] - prediction[:, :, 1:, :]
        vertical_target = target[:, :, :-1, :] - target[:, :, 1:, :]
        horizontal_prediction = prediction[:, :, :, :-1] - prediction[:, :, :, 1:]
        horizontal_target = target[:, :, :, :-1] - target[:, :, :, 1:]
        vertical = pointwise_distance(vertical_prediction, vertical_target, criterion=self.criterion)
        horizontal = pointwise_distance(horizontal_prediction, horizontal_target, criterion=self.criterion)
        return 0.5 * (vertical + horizontal)


@LOSS_REGISTRY.register("spectral_edge")
class SpectralEdgeRegularizationLoss(_TargetGradientRegularizationLoss):
    """Match adjacent-band spectral gradients of prediction and HRHS."""

    registry_name = "spectral_edge"

    def forward(self, prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        target = self.target(batch)
        if prediction.shape[1] < 2:
            raise ConfigurationError("spectral_edge requires at least two HSI channels.")
        prediction_gradient = prediction[:, :-1, :, :] - prediction[:, 1:, :, :]
        target_gradient = target[:, :-1, :, :] - target[:, 1:, :, :]
        return pointwise_distance(prediction_gradient, target_gradient, criterion=self.criterion)
