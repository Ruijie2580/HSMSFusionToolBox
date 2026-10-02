"""Shared in-memory HS–MS fusion DataModule template.

`BaseFusionDataModule` carries the common pipeline for all benchmark datasets:
HRHS loading with [0, 1]/finite validation, observation materialization,
train/valid/test dataloader construction, and the `num_workers=0` hard check.
Dataset subclasses declare their differences through the class attributes
`dataset_label` and `default_hsi_channels` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hsmsfusion.core.exceptions import ConfigurationError, ContractError
from hsmsfusion.core.protocols import validate_nchw
from hsmsfusion.degradations import build_observation_generator


class InMemoryFusionDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor-indexed fusion samples with an optional supervised target view."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        expose_target: bool,
    ) -> None:
        self.tensors = dict(tensors)
        self.expose_target = expose_target
        lengths = {tensor.shape[0] for tensor in self.tensors.values()}
        if len(lengths) != 1:
            raise ContractError("All in-memory fusion tensors must share a sample count.")

    def __len__(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "lr_hsi": self.tensors["lr_hsi"][index],
            "hr_msi": self.tensors["hr_msi"][index],
        }
        if self.expose_target:
            sample["gt_hsi"] = self.tensors["gt_hsi"][index]
        return sample


class BaseFusionDataModule(LightningDataModule):
    """Load HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label: str
    default_hsi_channels: int

    def __init__(
        self,
        dataset_config: Mapping[str, object],
        degradation_config: Mapping[str, object],
        dataloader_config: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.dataset_config = dict(dataset_config)
        self.degradation_config = dict(degradation_config)
        self.dataloader_config = dict(dataloader_config)
        self.root = Path(str(self.dataset_config["root"])).expanduser()
        self.materialization_batch_size = int(
            self.dataloader_config["materialization_batch_size"]
        )
        self.expected_hsi_channels = int(
            self.dataset_config.get("hsi_channels", self.default_hsi_channels)
        )
        if str(self.dataset_config.get("cache_device", "cpu")).lower() != "cpu":
            raise ConfigurationError(
                f"In-memory {self.dataset_label} currently supports cache_device=cpu only."
            )
        self._loader_options = {
            split: self._validate_loader_options(split)
            for split in ("train", "valid", "test")
        }
        self.observation_generator = build_observation_generator(self.degradation_config)
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._train_expose_target = True

    def configure_model_visibility(self, *, requires_gt: bool) -> None:
        """Hide HRHS from unsupervised training batches."""
        self._train_expose_target = requires_gt

    def _load_hrhs(self, split: str) -> torch.Tensor:
        file_path = self.root / f"{split}.h5"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing {self.dataset_label} split: {file_path}")
        with h5py.File(file_path, "r") as handle:
            if "HRHS" not in handle:
                raise ContractError(f"{file_path} must contain the HRHS dataset.")
            array = np.asarray(handle["HRHS"], dtype=np.float32)
        tensor = torch.from_numpy(array).contiguous()
        validate_nchw(tensor, name=f"{split}.HRHS", channels=self.expected_hsi_channels)
        if not torch.isfinite(tensor).all():
            raise ContractError(f"{split}.HRHS contains NaN or Inf.")
        minimum = float(tensor.min())
        maximum = float(tensor.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ContractError(
                f"{split}.HRHS must be normalized to [0, 1], got [{minimum}, {maximum}]."
            )
        return tensor

    def _prepare_split(self, split: str) -> None:
        if split in self._cache:
            return
        hr_hsi = self._load_hrhs(split)
        observations = self.observation_generator.generate_in_batches(
            hr_hsi,
            batch_size=self.materialization_batch_size,
        )
        self._cache[split] = {
            "gt_hsi": hr_hsi,
            "hr_msi": observations["hr_msi"],
            "lr_hsi": observations["lr_hsi"],
        }

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"}:
            self._prepare_split("train")
            self._prepare_split("valid")
        if stage in {None, "test", "predict"}:
            self._prepare_split("test")

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._make_dataloader("train", expose_target=self._train_expose_target)

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._make_dataloader("valid", expose_target=True)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._make_dataloader("test", expose_target=True)

    def predict_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self.test_dataloader()

    def _validate_loader_options(self, split: str) -> dict[str, object]:
        if split not in self.dataloader_config:
            raise ConfigurationError(f"Missing dataloader configuration for split {split!r}.")
        options = dict(self.dataloader_config[split])
        batch_size = int(options.get("batch_size", 0))
        if batch_size <= 0:
            raise ConfigurationError(f"{split} dataloader batch_size must be positive.")
        num_workers = int(options.get("num_workers", 0))
        if num_workers != 0:
            raise ConfigurationError(
                f"In-memory {self.dataset_label} requires num_workers=0 to prevent "
                "Windows dataset copies."
            )
        return {
            "batch_size": batch_size,
            "shuffle": bool(options.get("shuffle", False)),
            "pin_memory": bool(options.get("pin_memory", True)),
            "num_workers": num_workers,
        }

    def _make_dataloader(
        self,
        split: str,
        *,
        expose_target: bool,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        options = self._loader_options[split]
        return DataLoader(
            InMemoryFusionDataset(self._cache[split], expose_target=expose_target),
            batch_size=int(options["batch_size"]),
            shuffle=bool(options["shuffle"]),
            num_workers=int(options["num_workers"]),
            pin_memory=bool(options["pin_memory"]),
            persistent_workers=False,
        )
