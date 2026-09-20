"""Smoke tests for the SSIL-Net port."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hsmsfusion.models.SSILNet.lightning_module import SSILNetLightningModule
from hsmsfusion.models.SSILNet.network import FuseBlock7, Multi_Scale, ResSPA, SSILNet


def _tiny_network_kwargs() -> dict:
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
        "hidden": 8,
    }


def test_ssilnet_forward_shape() -> None:
    torch.manual_seed(0)
    model = SSILNet(hsi_channels=4, msi_channels=2, scale=4)
    out = model(torch.rand(2, 4, 16, 16), torch.rand(2, 2, 64, 64))
    assert out.shape == (2, 4, 64, 64)
    assert torch.isfinite(out).all()


def test_ssilnet_architecture_keeps_upstream_width_and_depth() -> None:
    """Only scale factors and input channels may differ from upstream."""
    model = SSILNet(hsi_channels=128, msi_channels=8, scale=4)

    # Width: the hidden width is upstream's 64, not rescaled.
    assert model.hidden == 64
    # Depth: upstream's block counts, unchanged.
    assert sum(isinstance(m, ResSPA) for m in model.modules()) == 3
    assert sum(isinstance(m, FuseBlock7) for m in model.modules()) == 4
    assert sum(isinstance(m, Multi_Scale) for m in model.modules()) == 3
    # The ladder's two steps multiply to the recipe's scale.
    assert model.hsi_to_msi * model.msi_to_pan == model.scale


def test_ssilnet_ladder_follows_the_configured_scale() -> None:
    """The HSI-side factor tracks ``scale``; the MSI side is a no-op."""
    model = SSILNet(hsi_channels=8, msi_channels=2, scale=2)
    assert model.hsi_to_msi == 2
    assert model.msi_to_pan == 1
    assert model.srh4.pixel_shuffle.upscale_factor == 2
    assert model.up_hsi_pan.scale_factor == 2
    assert model.up_hsi_msi.scale_factor == 2

    out = model(torch.rand(1, 8, 16, 16), torch.rand(1, 2, 32, 32))
    assert out.shape == (1, 8, 32, 32)


def test_ssilnet_msi_side_factors_are_identity_and_same_resolution() -> None:
    """The collapsed top step must not upsample, but must still compute features."""
    model = SSILNet(**_tiny_network_kwargs())
    probe = torch.rand(1, 2, 32, 32)
    assert torch.allclose(model.up_msi_pan(probe), probe, atol=1e-6)

    # SuperResolution(1) is a real conv stack, not a passthrough.
    assert model.srm4.pixel_shuffle.upscale_factor == 1
    assert model.srm4(probe).shape == probe.shape
    assert not torch.allclose(model.srm4(probe), probe)


def test_ssilnet_pan_guide_is_derived_from_the_principle_observation() -> None:
    """The panchromatic guide is the HRMSI band-mean, never a reference statistic."""
    torch.manual_seed(0)
    model = SSILNet(**_tiny_network_kwargs()).eval()
    lr_hsi = torch.rand(1, 4, 16, 16)
    hr_msi = torch.rand(1, 2, 64, 64)
    with torch.no_grad():
        baseline = model(lr_hsi, hr_msi)
        # Perturbing only the HRMSI must move the output through the pan path.
        perturbed = model(lr_hsi, hr_msi + 0.5)
    assert not torch.allclose(baseline, perturbed)


def test_ssilnet_predicts_a_residual_over_the_upsampled_lr_hsi() -> None:
    torch.manual_seed(0)
    model = SSILNet(**_tiny_network_kwargs())
    with torch.no_grad():
        model.last_conv[1].weight.zero_()
        model.last_conv[1].bias.zero_()
        lr_hsi = torch.rand(1, 4, 16, 16)
        out = model(lr_hsi, torch.rand(1, 2, 64, 64))
    baseline = torch.nn.functional.interpolate(
        lr_hsi, scale_factor=4, mode="bicubic", align_corners=True
    )
    assert torch.allclose(out, baseline, atol=1e-6)


def test_ssilnet_rejects_inputs_that_miss_the_ladder() -> None:
    model = SSILNet(**_tiny_network_kwargs()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should be"):
        model(torch.rand(1, 4, 8, 8), torch.rand(1, 2, 64, 64))
    with torch.no_grad(), pytest.raises(ValueError, match="divisible by scale"):
        model(torch.rand(1, 4, 16, 16), torch.rand(1, 2, 63, 64))


def test_ssilnet_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = SSILNetLightningModule(
        model_config={
            "name": "SSILNet",
            "params": _tiny_network_kwargs(),
        },
        optimizer_config={"name": "adam", "lr": 1e-3},
        scheduler_config={"name": "none"},
        loss_config={
            "reconstruction": [{"name": "l1", "alias": "reconstruction", "weight": 1.0}],
            "regularization": [{"name": "sam", "alias": "spectral_angle", "weight": 0.1 / math.pi}],
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


def test_ssilnet_recipe_composes() -> None:
    from hydra import compose, initialize_config_dir
    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/ssilnet_x4"])

    assert config.experiment.name == "ssilnet_chikusei_x4"
    assert config.model.name == "SSILNet"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.model.params.hidden == 64
    assert config.model.inference.mode == "full"
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200

    # Upstream's L1 + 0.1 * (radians / pi) becomes L1 + (0.1 / pi) * radians.
    terms = {term.alias: term for term in config.training.loss.regularization}
    assert terms["spectral_angle"].name == "sam"
    assert terms["spectral_angle"].weight == pytest.approx(0.1 / math.pi, rel=1e-6)

    model = _build_model(config)
    assert type(model).__name__ == "SSILNetLightningModule"
    assert type(model.network).__name__ == "SSILNet"
