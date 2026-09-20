from __future__ import annotations

import h5py
import numpy as np

from hsmsfusion.data.datasets.icvl.datamodule import ICVLDataModule


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


def test_icvl_training_view_and_materialized_observations(tmp_path) -> None:
    for split, count in (("train", 3), ("valid", 2), ("test", 1)):
        _write_split(tmp_path, split, count)
    datamodule = ICVLDataModule(
        {
            "root": str(tmp_path),
            "hsi_channels": 4,
        },
        _degradation_config(),
        {
            "materialization_batch_size": 2,
            "train": {"batch_size": 2, "shuffle": True, "pin_memory": False, "num_workers": 0},
            "valid": {"batch_size": 1, "shuffle": False, "pin_memory": False, "num_workers": 0},
            "test": {"batch_size": 1, "shuffle": False, "pin_memory": False, "num_workers": 0},
        },
    )
    datamodule.configure_model_visibility(requires_gt=False)
    datamodule.setup("fit")

    train_batch = next(iter(datamodule.train_dataloader()))
    validation_batch = next(iter(datamodule.val_dataloader()))
    assert "gt_hsi" not in train_batch
    assert validation_batch["gt_hsi"].shape[1:] == (4, 8, 8)
    assert validation_batch["gt_hsi"].shape[0] == 1
    assert train_batch["lr_hsi"].shape[-2:] == (4, 4)
    assert train_batch["hr_msi"].shape[1:] == (2, 8, 8)
