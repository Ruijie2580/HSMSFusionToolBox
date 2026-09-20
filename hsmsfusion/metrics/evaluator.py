"""Independent CPU float32 evaluator and stable result-file writers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from hsmsfusion.core.exceptions import ContractError
from hsmsfusion.core.registry import METRIC_REGISTRY
from hsmsfusion.metrics import implementations as _implementations  # noqa: F401


@dataclass
class EvaluationResult:
    """Per-image and aggregate metrics for one benchmark run."""

    per_image: list[dict[str, float | int]]
    summary: dict[str, object]


class Evaluator:
    """Reference evaluator shared by validation and offline prediction scoring."""

    def __init__(self, metric_config: Mapping[str, object], *, scale: int) -> None:
        self.metric_config = dict(metric_config)
        self.scale = int(scale)
        self.leaderboard = list(self.metric_config["leaderboard"])
        self.context: dict[str, object] = {
            "scale": self.scale,
            "epsilon": float(self.metric_config.get("epsilon", 1e-8)),
            "sam_epsilon": float(self.metric_config.get("sam_epsilon", 1e-8)),
            "ssim": dict(self.metric_config.get("ssim", {})),
        }
        for metric_name in self.leaderboard:
            METRIC_REGISTRY.get(metric_name)

    @staticmethod
    def _validate_array(array: np.ndarray, *, name: str) -> None:
        if array.ndim != 4:
            raise ContractError(f"{name} must use NCHW layout, got {array.shape}.")
        if array.dtype != np.float32:
            raise ContractError(f"{name} must be float32, got {array.dtype}.")
        if not np.isfinite(array).all():
            raise ContractError(f"{name} contains NaN or Inf.")

    def evaluate(self, prediction: np.ndarray, target: np.ndarray) -> EvaluationResult:
        self._validate_array(prediction, name="prediction")
        self._validate_array(target, name="target")
        if prediction.shape != target.shape:
            raise ContractError(
                f"prediction and target shape mismatch: {prediction.shape} vs {target.shape}."
            )
        prediction = np.clip(prediction, 0.0, 1.0)
        per_image: list[dict[str, float | int]] = []
        for index in range(prediction.shape[0]):
            estimated = torch.from_numpy(prediction[index])
            reference = torch.from_numpy(target[index])
            row: dict[str, float | int] = {"index": index}
            for metric_name in self.leaderboard:
                metric = METRIC_REGISTRY.get(metric_name)
                row[metric_name] = metric(estimated, reference, self.context)
            row["sam_rad"] = METRIC_REGISTRY.get("sam_rad")(
                estimated, reference, self.context
            )
            per_image.append(row)

        summary_metrics: dict[str, dict[str, float]] = {}
        for metric_name in self.leaderboard:
            values = np.asarray([float(row[metric_name]) for row in per_image], dtype=np.float32)
            summary_metrics[metric_name] = {
                "mean": float(values.mean(dtype=np.float32)),
                "std": float(values.std(dtype=np.float32, ddof=0)),
            }
        return EvaluationResult(
            per_image=per_image,
            summary={
                "metric_protocol": self.metric_config.get("protocol_version", "hif_metrics_v1"),
                "aggregation": "per_image_mean_population_std",
                "std_type": "population",
                "ddof": 0,
                "scale": self.scale,
                "metrics": summary_metrics,
            },
        )


def load_prediction_file(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"prediction"}:
            raise ContractError("predictions.h5 must contain only the /prediction dataset.")
        return np.asarray(handle["prediction"], dtype=np.float32)


def write_evaluation_result(result: EvaluationResult, output_directory: Path) -> None:
    """Write the complete benchmark report as one human-readable CSV file."""

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = list(result.summary["metrics"])
    headers = ["Image", *[_metric_header(metric_name) for metric_name in metrics]]
    metrics_path = output_directory / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index, row in enumerate(result.per_image):
            writer.writerow(
                [f"image_{index:04d}", *[_format_metric(float(row[name])) for name in metrics]]
            )
        aggregate = result.summary["metrics"]
        writer.writerow(
            ["Average", *[_format_metric(float(aggregate[name]["mean"])) for name in metrics]]
        )
        writer.writerow(
            ["Std", *[_format_metric(float(aggregate[name]["std"])) for name in metrics]]
        )


def _metric_header(metric_name: str) -> str:
    headers = {
        "psnr_global": "PSNR",
        "ssim_band_mean": "SSIM",
        "sam_deg": "*SAM",
        "ergas": "*ERGAS",
        "rmse_global": "RMSE",
        "uiqi": "UIQI",
    }
    return headers.get(metric_name, metric_name)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def evaluate_prediction_file(
    *,
    prediction_path: Path,
    target_path: Path,
    metric_config: Mapping[str, object],
    scale: int,
    output_directory: Path,
) -> EvaluationResult:
    prediction = load_prediction_file(prediction_path)
    with h5py.File(target_path, "r") as handle:
        if "HRHS" not in handle:
            raise ContractError(f"{target_path} does not contain HRHS.")
        target = np.asarray(handle["HRHS"], dtype=np.float32)
    # Handle NHWC → NCHW transpose (e.g. Harvard dataset).
    if target.ndim == 4 and target.shape[-1] == prediction.shape[1] and target.shape[1] != prediction.shape[1]:
        target = np.transpose(target, (0, 3, 1, 2)).astype(np.float32, copy=False)
    result = Evaluator(metric_config, scale=scale).evaluate(prediction, target)
    write_evaluation_result(result, output_directory)
    return result
