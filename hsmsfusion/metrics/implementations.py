"""CPU float32 reference metrics for the HIF benchmark protocol."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import METRIC_REGISTRY


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


@METRIC_REGISTRY.register("psnr_global")
def psnr_global(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    epsilon = float(context.get("epsilon", 1e-8))
    mse = torch.mean((prediction - target) ** 2)
    return _scalar(10.0 * torch.log10(torch.tensor(1.0, dtype=torch.float32) / mse.clamp_min(epsilon)))


@METRIC_REGISTRY.register("rmse_global")
def rmse_global(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    return _scalar(torch.sqrt(torch.mean((prediction - target) ** 2)))


def per_band_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> torch.Tensor:
    epsilon = float(context.get("epsilon", 1e-8))
    mse = torch.mean((prediction - target) ** 2, dim=(-2, -1))
    return 10.0 * torch.log10(torch.ones_like(mse) / mse.clamp_min(epsilon))


@METRIC_REGISTRY.register("ssim_band_mean")
def ssim_band_mean(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    """Compute 2-D SSIM independently for each HSI band, then average bands."""
    config = context.get("ssim", {})
    window_size = int(config.get("window_size", 11))
    sigma = float(config.get("sigma", 1.5))
    padding_mode = str(config.get("padding", "reflect"))
    if window_size < 1 or window_size % 2 == 0:
        raise ConfigurationError("SSIM window_size must be a positive odd integer.")
    if sigma <= 0:
        raise ConfigurationError("SSIM sigma must be positive.")
    if prediction.shape[-2] <= window_size // 2 or prediction.shape[-1] <= window_size // 2:
        raise ConfigurationError("SSIM window is too large for the evaluated image.")

    coordinate = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    window_1d = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    window_1d /= window_1d.sum()
    window = torch.outer(window_1d, window_1d)[None, None]
    channels = prediction.shape[0]
    window = window.expand(channels, 1, -1, -1)
    padding = window_size // 2
    prediction_4d = prediction[None]
    target_4d = target[None]
    if padding_mode == "zeros":
        def convolve(tensor: torch.Tensor) -> torch.Tensor:
            return functional.conv2d(tensor, window, padding=padding, groups=channels)
    elif padding_mode in {"reflect", "replicate", "circular"}:
        def convolve(tensor: torch.Tensor) -> torch.Tensor:
            padded = functional.pad(
                tensor, (padding, padding, padding, padding), mode=padding_mode
            )
            return functional.conv2d(padded, window, groups=channels)

    else:
        raise ConfigurationError(f"Unsupported SSIM padding mode: {padding_mode!r}.")

    mu_prediction = convolve(prediction_4d)
    mu_target = convolve(target_4d)
    variance_prediction = convolve(prediction_4d**2) - mu_prediction**2
    variance_target = convolve(target_4d**2) - mu_target**2
    covariance = convolve(prediction_4d * target_4d) - mu_prediction * mu_target
    c1 = float(config.get("k1", 0.01)) ** 2
    c2 = float(config.get("k2", 0.03)) ** 2
    numerator = (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
    denominator = (mu_prediction**2 + mu_target**2 + c1) * (
        variance_prediction + variance_target + c2
    )
    score_per_band = (numerator / denominator.clamp_min(1e-12)).mean(dim=(-2, -1))
    return _scalar(score_per_band.mean())


def _sam_rad_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> torch.Tensor:
    epsilon = float(context.get("sam_epsilon", 1e-8))
    prediction_pixels = prediction.flatten(start_dim=1)
    target_pixels = target.flatten(start_dim=1)
    numerator = (prediction_pixels * target_pixels).sum(dim=0)
    denominator = prediction_pixels.norm(dim=0) * target_pixels.norm(dim=0) + epsilon
    cosine = (numerator / denominator).clamp(-1.0, 1.0)
    return torch.arccos(cosine).mean()


@METRIC_REGISTRY.register("sam_rad")
def sam_rad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    return _scalar(_sam_rad_tensor(prediction, target, context))


@METRIC_REGISTRY.register("sam_deg")
def sam_deg(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    radians = _sam_rad_tensor(prediction, target, context)
    return _scalar(radians * (180.0 / torch.pi))


@METRIC_REGISTRY.register("ergas")
def ergas(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    scale = int(context["scale"])
    epsilon = float(context.get("epsilon", 1e-8))
    if scale < 1:
        raise ConfigurationError("ERGAS scale must be positive.")
    rmse_per_band = torch.sqrt(torch.mean((prediction - target) ** 2, dim=(-2, -1)))
    reference_mean = target.mean(dim=(-2, -1))
    value = (100.0 / scale) * torch.sqrt(
        torch.mean((rmse_per_band / (reference_mean + epsilon)) ** 2)
    )
    return _scalar(value)


@METRIC_REGISTRY.register("uiqi")
def uiqi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: Mapping[str, object],
) -> float:
    """Universal Image Quality Index (UIQI), computed globally per band then averaged."""
    epsilon = float(context.get("epsilon", 1e-8))
    # prediction/target: (C, H, W) — spatial dims flattened
    pred_flat = prediction.flatten(start_dim=1)  # (C, H*W)
    tgt_flat = target.flatten(start_dim=1)

    mu_p = pred_flat.mean(dim=1)   # (C,)
    mu_t = tgt_flat.mean(dim=1)

    sigma_p2 = pred_flat.var(dim=1, unbiased=False)
    sigma_t2 = tgt_flat.var(dim=1, unbiased=False)
    sigma_pt = ((pred_flat - mu_p.unsqueeze(1)) * (tgt_flat - mu_t.unsqueeze(1))).mean(dim=1)

    numerator = 4.0 * sigma_pt * mu_p * mu_t
    denominator = (sigma_p2 + sigma_t2) * (mu_p ** 2 + mu_t ** 2)
    q_per_band = numerator / denominator.clamp_min(epsilon)
    return _scalar(q_per_band.mean())


def register_default_metrics() -> None:
    """Import side effects register metrics; retained as an explicit public hook."""
    return None

