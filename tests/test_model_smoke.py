from __future__ import annotations

from types import SimpleNamespace

import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from hsmsfusion.callbacks import EpochMetricReporter, EpochRichProgressCallback, RNGStateCallback
from hsmsfusion.cli import _last_checkpoint_callback
from hsmsfusion.models.SMGU.lightning_module import SMGULightningModule


class _TensorFusionDataset(Dataset):
    def __init__(self, count: int) -> None:
        generator = torch.Generator().manual_seed(count)
        self.gt = torch.rand(count, 4, 8, 8, generator=generator)
        self.msi = torch.rand(count, 2, 8, 8, generator=generator)
        self.lr = torch.rand(count, 4, 2, 2, generator=generator)

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, index: int):
        return {"gt_hsi": self.gt[index], "hr_msi": self.msi[index], "lr_hsi": self.lr[index]}


class _SmokeDataModule(LightningDataModule):
    def setup(self, stage=None) -> None:
        self.train_data = _TensorFusionDataset(3)
        self.valid_data = _TensorFusionDataset(2)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=1, shuffle=False)


def _model() -> SMGULightningModule:
    return SMGULightningModule(
        model_config={
            "name": "SMGU",
            "params": {"hsi_channels": 4, "msi_channels": 2, "scale": 4, "niter": 1},
        },
        optimizer_config={"name": "adam", "lr": 1e-3},
        scheduler_config={"name": "none"},
        loss_config={
            "reconstruction": [{"name": "l1", "alias": "reconstruction", "weight": 1.0}],
            "regularization": [],
        },
        metric_config={
            "leaderboard": ["psnr_global", "ssim_band_mean", "sam_deg", "ergas", "rmse_global", "uiqi"],
            "ssim": {"window_size": 3, "sigma": 1.0, "padding": "reflect"},
        },
        scale=4,
    )


def test_model_smoke_contract_and_checkpoint_restore(tmp_path) -> None:
    callback = ModelCheckpoint(
        dirpath=tmp_path,
        filename="best",
        monitor="val/psnr_global",
        mode="max",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        callbacks=[callback, RNGStateCallback()],
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    datamodule = _SmokeDataModule()
    model = _model()
    trainer.fit(model, datamodule=datamodule)

    assert (tmp_path / "last.ckpt").is_file()
    assert (tmp_path / "best.ckpt").is_file()
    safe_checkpoint = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=True)
    assert "hsmsfusion_rng_state" in safe_checkpoint
    restored = _model()
    state = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    restored.load_state_dict(state["state_dict"])
    output = restored(torch.rand(1, 4, 2, 2), torch.rand(1, 2, 8, 8))
    assert output.shape == (1, 4, 8, 8)

    resume_trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        callbacks=[RNGStateCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    resume_trainer.fit(_model(), datamodule=datamodule, ckpt_path=tmp_path / "last.ckpt")


def test_multistep_scheduler_is_constructed_with_epoch_interval() -> None:
    model = _model()
    model.scheduler_config = {"name": "MultiStepLR", "milestones": [2, 4], "gamma": 0.5}

    configured = model.configure_optimizers()

    scheduler_config = configured["lr_scheduler"]
    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["scheduler"].milestones[2] == 1
    assert scheduler_config["scheduler"].milestones[4] == 1
    assert scheduler_config["scheduler"].gamma == 0.5


def test_rich_epoch_progress_supports_multiple_epochs() -> None:
    """Rich task IDs are opaque and must not be treated as list indexes."""
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        callbacks=[EpochRichProgressCallback(), EpochMetricReporter()],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    trainer.fit(_model(), datamodule=_SmokeDataModule())
    assert "val/loss" in trainer.callback_metrics


def test_epoch_reporter_does_not_reuse_stale_validation_metrics() -> None:
    reporter = EpochMetricReporter()
    captured: list[dict[str, object]] = []
    reporter._print_epoch_summary = lambda **summary: captured.append(summary)  # type: ignore[method-assign]
    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={
            "train/loss": torch.tensor(0.1),
            "val/psnr_global": torch.tensor(33.0),
        },
        current_epoch=3,
        max_epochs=200,
    )

    reporter.on_fit_start(trainer, pl_module=None)
    reporter.on_train_epoch_start(trainer, pl_module=None)
    reporter.on_train_epoch_end(trainer, pl_module=None)

    assert captured[-1]["validation_metrics"] == {}

    reporter.on_train_epoch_start(trainer, pl_module=None)
    reporter.on_validation_start(trainer, pl_module=None)
    reporter.on_validation_end(trainer, pl_module=None)
    reporter.on_train_epoch_end(trainer, pl_module=None)

    assert captured[-1]["validation_metrics"] == {"PSNR": ("33.00000000", "bright_green")}


def test_epoch_reporter_lists_each_configured_training_loss_term() -> None:
    reporter = EpochMetricReporter()
    captured: list[dict[str, object]] = []
    reporter._print_epoch_summary = lambda **summary: captured.append(summary)  # type: ignore[method-assign]
    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={
            "train/loss": torch.tensor(0.3),
            "train/reconstruction": torch.tensor(0.2),
            "train/reconstruction_weighted": torch.tensor(0.2),
            "train/spectral_angle": torch.tensor(0.1),
            "train/spectral_angle_weighted": torch.tensor(0.1),
        },
        current_epoch=0,
        max_epochs=1,
    )
    module = SimpleNamespace(
        loss_manager=SimpleNamespace(
            components=(
                SimpleNamespace(alias="reconstruction"),
                SimpleNamespace(alias="spectral_angle"),
            )
        )
    )

    reporter.on_fit_start(trainer, pl_module=module)
    reporter.on_train_epoch_start(trainer, pl_module=module)
    reporter.on_train_epoch_end(trainer, pl_module=module)

    assert captured[-1]["training_loss_terms"] == [
        ("reconstruction", "0.20000000", "0.20000000"),
        ("spectral_angle", "0.10000000", "0.10000000"),
    ]


def test_last_checkpoint_is_saved_without_a_validation_pass(tmp_path) -> None:
    checkpoint_callback = _last_checkpoint_callback(tmp_path)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        check_val_every_n_epoch=2,
        logger=False,
        callbacks=[checkpoint_callback, RNGStateCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(_model(), datamodule=_SmokeDataModule())

    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "hsmsfusion_rng_state" in checkpoint

    resume_trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        check_val_every_n_epoch=2,
        logger=False,
        callbacks=[_last_checkpoint_callback(tmp_path), RNGStateCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    resume_trainer.fit(_model(), datamodule=_SmokeDataModule(), ckpt_path=checkpoint_path)

    assert resume_trainer.global_step == 2
