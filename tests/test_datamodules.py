"""Parametrized DataModule contracts covering every registered dataset."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.discovery import discover_datasets

discover_datasets()

_DATASET_NAMES = DATASET_REGISTRY.names()


def _write_split(root, split: str, count: int) -> None:
    array = np.random.default_rng(count).random((count, 4, 8, 8), dtype=np.float32)
    with h5py.File(root / f"{split}.h5", "w") as handle:
        handle.create_dataset("HRHS", data=array)


def _degradation_config() -> dict[str, object]:
    return {
        "spectral": {
            "name": "gaussian_srf",
            "wavelength_start_nm": 400.0,
            "wavelength_end_nm": 700.0,
            "num_hsi_bands": 4,
            "centers_nm": [450.0, 650.0],
            "fwhm_nm": [80.0, 80.0],
        },
        "spatial": {
            "name": "gaussian_decimation",
            "scale": 2,
            "kernel_size": 3,
            "sigma": 1.0,
            "padding": "reflect",
            "offset": [0, 0],
        },
    }


def _dataloader_config(num_workers: int = 0) -> dict[str, object]:
    return {
        "materialization_batch_size": 2,
        "train": {"batch_size": 2, "shuffle": True, "pin_memory": False, "num_workers": num_workers},
        "valid": {"batch_size": 1, "shuffle": False, "pin_memory": False, "num_workers": num_workers},
        "test": {"batch_size": 1, "shuffle": False, "pin_memory": False, "num_workers": num_workers},
    }


def _build_datamodule(dataset_name: str, root, num_workers: int = 0):
    datamodule_class = DATASET_REGISTRY.get(dataset_name)
    return datamodule_class(
        {"root": str(root), "hsi_channels": 4},
        _degradation_config(),
        _dataloader_config(num_workers=num_workers),
    )


@pytest.mark.parametrize("dataset_name", _DATASET_NAMES)
def test_training_view_and_materialized_observations(tmp_path, dataset_name: str) -> None:
    for split, count in (("train", 3), ("valid", 2), ("test", 1)):
        _write_split(tmp_path, split, count)
    datamodule = _build_datamodule(dataset_name, tmp_path)
    datamodule.configure_model_visibility(requires_gt=False)
    datamodule.setup("fit")

    train_batch = next(iter(datamodule.train_dataloader()))
    validation_batch = next(iter(datamodule.val_dataloader()))
    assert "gt_hsi" not in train_batch
    assert "index" in train_batch
    assert validation_batch["gt_hsi"].shape[1:] == (4, 8, 8)
    assert validation_batch["gt_hsi"].shape[0] == 1
    assert train_batch["lr_hsi"].shape[-2:] == (4, 4)
    assert train_batch["hr_msi"].shape[1:] == (2, 8, 8)


@pytest.mark.parametrize("dataset_name", _DATASET_NAMES)
def test_nonzero_num_workers_rejected(tmp_path, dataset_name: str) -> None:
    with pytest.raises(ConfigurationError, match="num_workers=0"):
        _build_datamodule(dataset_name, tmp_path, num_workers=1)
