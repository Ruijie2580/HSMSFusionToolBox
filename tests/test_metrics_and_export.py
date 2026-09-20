from __future__ import annotations

import csv

import h5py
import numpy as np

from hsmsfusion.evaluation import write_prediction_file
from hsmsfusion.metrics.evaluator import Evaluator, load_prediction_file, write_evaluation_result
from hsmsfusion.utils.experiment import benchmark_output_directory


def _metric_config() -> dict[str, object]:
    return {
        "protocol_version": "test_v1",
        "leaderboard": [
            "psnr_global",
            "ssim_band_mean",
            "sam_deg",
            "ergas",
            "rmse_global",
            "uiqi",
        ],
        "epsilon": 1e-8,
        "sam_epsilon": 1e-8,
        "ssim": {"window_size": 3, "sigma": 1.0, "padding": "reflect"},
    }


def test_evaluator_reports_expected_perfect_reconstruction(tmp_path) -> None:
    target = np.full((2, 4, 8, 8), 0.5, dtype=np.float32)
    result = Evaluator(_metric_config(), scale=2).evaluate(target.copy(), target)

    assert result.summary["ddof"] == 0
    assert result.summary["metrics"]["rmse_global"]["mean"] == 0.0
    assert result.summary["metrics"]["psnr_global"]["mean"] > 70.0

    write_evaluation_result(result, tmp_path)
    metrics_path = tmp_path / "metrics.csv"
    assert metrics_path.is_file()
    assert {path.name for path in tmp_path.iterdir()} == {"metrics.csv"}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Image", "PSNR", "SSIM", "*SAM", "*ERGAS", "RMSE", "UIQI"]
    assert rows[1][0] == "image_0000"
    assert rows[-2][0] == "Average"
    assert rows[-1][0] == "Std"


def test_prediction_file_contains_only_prediction_dataset(tmp_path) -> None:
    prediction = np.random.default_rng(0).random((2, 4, 8, 8), dtype=np.float32)
    path = tmp_path / "predictions.h5"
    write_prediction_file(path, prediction)

    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == {"prediction"}
        assert handle["prediction"].shape == prediction.shape
        assert handle["prediction"].dtype == np.dtype("float32")
    assert np.array_equal(load_prediction_file(path), prediction)


def test_output_directory_is_grouped_by_dataset_scale_and_model(tmp_path) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path / "output")},
            "dataset": {"name": "chikusei"},
            "degradation": {"spatial": {"scale": 4}},
            "model": {"name": "MCTNet"},
        }
    )

    assert benchmark_output_directory(config) == (
        tmp_path / "output" / "chikusei_x4" / "mctnet"
    )
