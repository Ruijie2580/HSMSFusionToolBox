from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from hsmsfusion.callbacks import TrainingVisualizationCallback
from hsmsfusion.cli import _load_run_configuration
from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.degradations.spectral.gaussian_srf import GaussianSRF
from hsmsfusion.utils.experiment import (
    create_run_directory,
    experiment_directory,
    save_resolved_config,
)
from hsmsfusion.visualization import save_srf_curve, save_validation_comparison


def _spectral_operator() -> GaussianSRF:
    return GaussianSRF(
        {
            "centers_nm": [450.0, 650.0],
            "fwhm_nm": [40.0, 60.0],
            "num_hsi_bands": 4,
            "wavelength_start_nm": 400.0,
            "wavelength_end_nm": 700.0,
        }
    )


def test_training_artifacts_are_written_as_png(tmp_path) -> None:
    spectral_operator = _spectral_operator()
    srf_path = tmp_path / "figures" / "srf_curve.png"
    assert save_srf_curve(spectral_operator, srf_path)
    assert srf_path.is_file() and srf_path.stat().st_size > 0

    batch = {
        "gt_hsi": torch.rand(16, 4, 8, 8),
        "lr_hsi": torch.rand(16, 4, 4, 4),
        "hr_msi": torch.rand(16, 2, 8, 8),
    }
    validation_path = tmp_path / "figures" / "validation" / "epoch_0001.png"
    save_validation_comparison(
        batch,
        torch.rand(16, 4, 8, 8),
        validation_path,
        spectral_operator=spectral_operator,
    )
    assert validation_path.is_file() and validation_path.stat().st_size > 0


def test_visualization_callback_writes_run_log_artifacts(tmp_path) -> None:
    spectral_operator = _spectral_operator()
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            observation_generator=SimpleNamespace(spectral_operator=spectral_operator)
        ),
        sanity_checking=False,
        current_epoch=0,
    )
    batch = {
        "gt_hsi": torch.rand(1, 4, 8, 8),
        "lr_hsi": torch.rand(1, 4, 4, 4),
        "hr_msi": torch.rand(1, 2, 8, 8),
    }
    callback = TrainingVisualizationCallback(tmp_path)
    callback.on_fit_start(trainer, pl_module=None)
    callback.on_validation_batch_end(
        trainer,
        pl_module=None,
        outputs={"prediction": torch.rand(1, 4, 8, 8)},
        batch=batch,
        batch_idx=0,
    )

    figures_directory = tmp_path / "logs" / "figures"
    assert (figures_directory / "srf_curve.png").is_file()
    assert (figures_directory / "validation" / "epoch_0001.png").is_file()


def test_non_resume_replaces_the_entire_matching_experiment_record(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    experiment_root = runs_root / "same_experiment"
    checkpoint_path = experiment_root / "checkpoints" / "last.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    (experiment_root / "logs" / "events.out.tfevents").parent.mkdir()
    (experiment_root / "logs" / "events.out.tfevents").write_text("events")
    (experiment_root / "predictions.h5").write_bytes(b"prediction")

    config = OmegaConf.create(
        {
            "paths": {"runs_dir": str(runs_root)},
            "experiment": {"name": "same_experiment"},
        }
    )
    run_directory, replaced = create_run_directory(config)

    assert run_directory == experiment_root.resolve()
    assert replaced is True
    assert list(run_directory.iterdir()) == []


def test_experiment_directory_rejects_paths_outside_the_runs_root(tmp_path) -> None:
    config = OmegaConf.create(
        {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "experiment": {"name": "../outside"},
        }
    )

    with pytest.raises(ConfigurationError, match="direct child"):
        experiment_directory(config)


def test_test_and_evaluate_resolve_the_unique_directory_from_experiment_name(tmp_path) -> None:
    config = OmegaConf.create(
        {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "experiment": {"name": "same_experiment"},
            "run_dir": None,
        }
    )
    run_directory = experiment_directory(config)
    run_directory.mkdir(parents=True)
    save_resolved_config(OmegaConf.create({"stored": "configuration"}), run_directory)

    resolved_directory, resolved_config = _load_run_configuration(config)

    assert resolved_directory == run_directory
    assert resolved_config.stored == "configuration"
