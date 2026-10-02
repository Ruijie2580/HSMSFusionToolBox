from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hsmsfusion.cli import (
    _build_datamodule,
    _build_model,
    _training_config,
    _validation_interval,
)
from hsmsfusion.core.exceptions import ConfigurationError


def _compose(*overrides: str):
    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        return compose(config_name="config", overrides=list(overrides))


def test_cyberfuse_recipe_owns_training_and_dataloader_configuration() -> None:
    config = _compose("experiment=chikusei/cyberfuse_x4")

    assert config.experiment.name == "cyberfuse_chikusei_x4"
    assert config.model.name == "CyberFuse"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.training.trainer.max_epochs == 300
    assert config.training.optimizer.lr == 2e-4
    assert config.training.scheduler.name == "MultiStepLR"
    assert list(config.training.scheduler.milestones) == [200, 250]
    assert "optimizer" not in config
    assert "trainer" not in config

    model = _build_model(config)
    datamodule = _build_datamodule(config, requires_gt=model.requires_gt)
    assert type(model).__name__ == "CyberFuseLightningModule"
    # Upstream declares `Net` twice in network.py; the second shadows the first.
    assert type(model.network).__name__ == "Net"
    assert datamodule._loader_options["valid"]["shuffle"] is False


def test_clsnet_recipe_owns_training_and_dataloader_configuration(require_recipe) -> None:
    require_recipe("chikusei/clsnet_x4")

    config = _compose("experiment=chikusei/clsnet_x4")

    assert config.experiment.name == "clsnet_chikusei_x4"
    assert config.model.name == "CLSNet"
    assert config.training.trainer.max_epochs == 200
    assert config.training.trainer.check_val_every_n_epoch == 5
    assert config.training.optimizer.lr == 1e-4
    assert config.training.dataloader.train.batch_size > 0
    assert config.training.dataloader.valid.shuffle is False
    assert "optimizer" not in config
    assert "trainer" not in config

    model = _build_model(config)
    datamodule = _build_datamodule(config, requires_gt=model.requires_gt)
    assert type(model).__name__ == "CLSNetLightningModule"
    assert type(model.network).__name__ == "CLSNet"
    assert datamodule._loader_options["valid"]["shuffle"] is False


def test_validation_cadence_is_recipe_owned_and_must_be_positive(require_recipe) -> None:
    require_recipe("chikusei/clsnet_x4")

    config = _compose(
        "experiment=chikusei/clsnet_x4",
        "training.trainer.check_val_every_n_epoch=5",
    )

    assert _validation_interval(config.training.trainer) == 5

    invalid_config = OmegaConf.create({"check_val_every_n_epoch": 0})
    with pytest.raises(ConfigurationError, match="positive integer"):
        _validation_interval(invalid_config)


def test_paviac_quickbird_recipe_composes_with_102_hsi_bands(require_recipe) -> None:
    require_recipe("paviac/cauwt_x4")

    config = _compose("experiment=paviac/cauwt_x4")

    assert config.experiment.name == "cauwt_paviac_x4"
    assert config.dataset.name == "paviac"
    assert config.dataset.hsi_channels == 102
    assert config.degradation.spectral.wavelength_start_nm == 430.0
    assert config.degradation.spectral.wavelength_end_nm == 840.0
    assert config.degradation.spectral.num_msi_bands == 4
    assert config.model.params.channel == 102
    assert config.model.params.ms_channel == 4

    model = _build_model(config)
    datamodule = _build_datamodule(config, requires_gt=model.requires_gt)
    assert type(datamodule).__name__ == "PaviaCDataModule"
    assert datamodule.observation_generator.spectral_operator.sensor == "QuickBird"
    assert datamodule.observation_generator.spectral_operator.srf.shape == (4, 102)


def test_icvl_nikon_recipe_composes_with_31_hsi_bands(require_recipe) -> None:
    require_recipe("icvl/cauwt_x4")

    config = _compose("experiment=icvl/cauwt_x4")

    assert config.experiment.name == "cauwt_icvl_x4"
    assert config.dataset.name == "icvl"
    assert config.dataset.hsi_channels == 31
    assert config.degradation.spectral.wavelength_start_nm == 400.0
    assert config.degradation.spectral.wavelength_end_nm == 700.0
    assert config.degradation.spectral.num_msi_bands == 3
    assert config.model.params.channel == 31
    assert config.model.params.ms_channel == 3
    assert config.model.inference.mode == "tiled"
    assert config.model.inference.tile_size == 256
    assert config.benchmark.name == "icvl_nikon_x4_known_v1"

    model = _build_model(config)
    datamodule = _build_datamodule(config, requires_gt=model.requires_gt)
    assert type(datamodule).__name__ == "ICVLDataModule"
    assert datamodule.observation_generator.spectral_operator.sensor == "Nikon_D700"
    assert datamodule.observation_generator.spectral_operator.srf.shape == (3, 31)


def test_archived_global_training_config_is_adapted_for_resume() -> None:
    legacy_config = OmegaConf.create(
        {
            "dataset": {
                "batch_size": 2,
                "generation_batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
            "trainer": {"max_epochs": 1, "log_every_n_steps": 1, "progress_bar": False},
            "optimizer": {"name": "adam", "lr": 1e-4},
            "scheduler": {"name": "none"},
            "loss": {"kind": "l1", "weight": 1.0},
            "checkpoint": {
                "monitor": "val/psnr_global",
                "mode": "max",
                "uses_validation_gt": True,
            },
        }
    )

    training = _training_config(legacy_config)
    assert training.dataloader.materialization_batch_size == 2
    assert training.dataloader.train.batch_size == 2
    assert training.dataloader.valid.shuffle is False
