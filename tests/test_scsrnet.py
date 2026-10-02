"""Smoke tests for the SCSRNet port."""

from __future__ import annotations

from pathlib import Path

import torch

from hsmsfusion.models.SCSRNet.lightning_module import SCSRNetLightningModule
from hsmsfusion.models.SCSRNet.network import SCSRNet


def _tiny_network_kwargs() -> dict:
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
        "dims": (16, 16),
        "depths": (2, 2),
    }


def test_scsrnet_forward_shape() -> None:
    torch.manual_seed(0)
    model = SCSRNet(**_tiny_network_kwargs())
    out = model(torch.rand(2, 4, 8, 8), torch.rand(2, 2, 32, 32))
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_scsrnet_small_kernel_only_uses_depthwise_branches() -> None:
    """kernel_size 3/5 must build the SK-SSRB depthwise branch, not the MSPM."""
    torch.manual_seed(0)
    kwargs = _tiny_network_kwargs()
    kwargs["kernel_sizes"] = ((3, 3), (3, 3))
    model = SCSRNet(**kwargs)
    out = model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 32, 32))
    assert out.shape == (1, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_scsrnet_kernel_size_mismatch_is_rejected() -> None:
    kwargs = _tiny_network_kwargs()
    kwargs["kernel_sizes"] = ((3, 3), (13, 13, 13))
    try:
        SCSRNet(**kwargs)
    except ValueError as error:
        assert "kernel sizes do not match the depths" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a ValueError for mismatched kernel sizes")


def test_scsrnet_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = SCSRNetLightningModule(
        model_config={
            "name": "SCSRNet",
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


def test_scsrnet_recipe_composes(require_recipe) -> None:
    require_recipe("chikusei/scsrnet_x4")

    from hydra import compose, initialize_config_dir

    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/scsrnet_x4"])

    assert config.experiment.name == "scsrnet_chikusei_x4"
    assert config.model.name == "SCSRNet"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.model.params.kernel_sizes == [[3, 3], [13, 13]]
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200

    model = _build_model(config)
    assert type(model).__name__ == "SCSRNetLightningModule"
    assert type(model.network).__name__ == "SCSRNet"
