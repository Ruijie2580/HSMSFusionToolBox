"""Smoke tests for the SatFusion HSMS adaptation."""

from __future__ import annotations

from pathlib import Path

import torch

from hsmsfusion.models.SatFusion.lightning_module import SatFusionLightningModule
from hsmsfusion.models.SatFusion.network import SatFusion


def _tiny_network_kwargs() -> dict:
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
        "hidden_channels": 16,
        "tr_dim": 8,
        "tr_depth": 2,
        "tr_heads": 2,
        "tr_mlp_dim": 16,
        "dropout": 0.0,
        "encoder_layers": 1,
        "pnn_n1": 8,
        "pnn_n2": 4,
        "use_msi_guide": True,
    }


def test_satfusion_forward_shape() -> None:
    torch.manual_seed(0)
    model = SatFusion(**_tiny_network_kwargs())
    lr_hsi = torch.rand(2, 4, 8, 8)
    hr_msi = torch.rand(2, 2, 32, 32)
    out = model(lr_hsi, hr_msi)
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_satfusion_without_msi_guide() -> None:
    torch.manual_seed(0)
    kwargs = _tiny_network_kwargs()
    kwargs["use_msi_guide"] = False
    model = SatFusion(**kwargs)
    out = model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 32, 32))
    assert out.shape == (1, 4, 32, 32)


def test_satfusion_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = SatFusionLightningModule(
        model_config={
            "name": "SatFusion",
            "params": _tiny_network_kwargs(),
        },
        optimizer_config={"name": "adam", "lr": 1e-3},
        scheduler_config={"name": "none"},
        loss_config={
            "reconstruction": [{"name": "l1", "alias": "reconstruction", "weight": 1.0}],
            "regularization": [],
        },
        metric_config={
            "leaderboard": ["psnr_global", "ssim_band_mean", "sam_deg", "ergas", "rmse_global"],
            "ssim": {"window_size": 3, "sigma": 1.0, "padding": "reflect"},
        },
        scale=4,
    )
    batch = {
        "lr_hsi": torch.rand(2, 4, 8, 8),
        "hr_msi": torch.rand(2, 2, 32, 32),
        "gt_hsi": torch.rand(2, 4, 32, 32),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_satfusion_recipe_composes() -> None:
    from hydra import compose, initialize_config_dir
    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/satfusion_x4"])

    assert config.experiment.name == "satfusion_chikusei_x4"
    assert config.model.name == "SatFusion"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200

    model = _build_model(config)
    assert type(model).__name__ == "SatFusionLightningModule"
    assert type(model.network).__name__ == "SatFusion"
