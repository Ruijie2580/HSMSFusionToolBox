"""Smoke tests for the CGSFN port."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hsmsfusion.models.CGSFN.lightning_module import CGSFNLightningModule
from hsmsfusion.models.CGSFN.network import CGSFN


def _tiny_network_kwargs() -> dict:
    # The trunk halves the feature map three times and the SFA window size is 8,
    # so the HR side has to be divisible by 64.
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
    }


def test_cgsfn_forward_shape() -> None:
    torch.manual_seed(0)
    model = CGSFN(**_tiny_network_kwargs())
    lr_hsi = torch.rand(2, 4, 16, 16)
    hr_msi = torch.rand(2, 2, 64, 64)
    out = model(lr_hsi, hr_msi)
    assert out.shape == (2, 4, 64, 64)
    assert torch.isfinite(out).all()


def test_cgsfn_scale_is_configurable() -> None:
    """The upstream x8 upsample must follow the recipe's scale."""
    torch.manual_seed(0)
    model = CGSFN(hsi_channels=4, msi_channels=2, scale=8)
    out = model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 64, 64))
    assert out.shape == (1, 4, 64, 64)
    assert torch.isfinite(out).all()


def test_cgsfn_predicts_a_residual_over_the_upsampled_lr_hsi() -> None:
    """With the tail zeroed the output collapses onto the bicubic baseline."""
    torch.manual_seed(0)
    model = CGSFN(**_tiny_network_kwargs())
    with torch.no_grad():
        model.tail[1].weight.zero_()
        model.tail[1].bias.zero_()
        lr_hsi = torch.rand(1, 4, 16, 16)
        hr_msi = torch.rand(1, 2, 64, 64)
        out = model(lr_hsi, hr_msi)
    baseline = torch.nn.functional.interpolate(
        lr_hsi, scale_factor=4, mode="bicubic", align_corners=False
    )
    assert torch.allclose(out, baseline, atol=1e-6)


def test_cgsfn_rejects_spatial_sizes_not_divisible_by_64() -> None:
    model = CGSFN(**_tiny_network_kwargs()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="divisible by 64"):
        model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 32, 32))


def test_cgsfn_rejects_non_square_input() -> None:
    """NWI adds a [B, C, 1, W] and a [B, C, W, 1] pool, so the map must be square."""
    model = CGSFN(**_tiny_network_kwargs()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="square HR input"):
        model(torch.rand(1, 4, 16, 16), torch.rand(1, 2, 64, 128))


def test_cgsfn_rejects_lr_hsi_inconsistent_with_the_scale() -> None:
    model = CGSFN(**_tiny_network_kwargs()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should be"):
        model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 64, 64))


def test_cgsfn_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = CGSFNLightningModule(
        model_config={
            "name": "CGSFN",
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
        "lr_hsi": torch.rand(2, 4, 16, 16),
        "hr_msi": torch.rand(2, 2, 64, 64),
        "gt_hsi": torch.rand(2, 4, 64, 64),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_cgsfn_tiled_inference_stitches_a_full_scene() -> None:
    torch.manual_seed(0)
    model = CGSFNLightningModule(
        model_config={
            "name": "CGSFN",
            "params": _tiny_network_kwargs(),
            "inference": {"mode": "tiled", "tile_size": 64, "shave": 16},
        },
        optimizer_config={"name": "adam", "lr": 1e-3},
        scheduler_config={"name": "none"},
        loss_config={
            "reconstruction": [{"name": "l1", "alias": "reconstruction", "weight": 1.0}],
            "regularization": [],
        },
        metric_config={
            "leaderboard": ["psnr_global"],
            "ssim": {"window_size": 3, "sigma": 1.0, "padding": "reflect"},
        },
        scale=4,
    )
    model.eval()
    inputs = {
        "lr_hsi": torch.rand(1, 4, 32, 32),
        "hr_msi": torch.rand(1, 2, 128, 128),
    }
    with torch.no_grad():
        prediction = model.predict_step(inputs, 0)
    assert prediction.shape == (1, 4, 128, 128)
    assert torch.isfinite(prediction).all()


def test_cgsfn_recipe_composes() -> None:
    from hydra import compose, initialize_config_dir
    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/cgsfn_x4"])

    assert config.experiment.name == "cgsfn_chikusei_x4"
    assert config.model.name == "CGSFN"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.model.inference.mode == "tiled"
    assert config.model.inference.tile_size == 64
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200

    model = _build_model(config)
    assert type(model).__name__ == "CGSFNLightningModule"
    assert type(model.network).__name__ == "CGSFN"
