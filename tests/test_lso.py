"""Smoke tests for the LSO port."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hsmsfusion.models.LSO.lightning_module import LSOLightningModule
from hsmsfusion.models.LSO.network import LSO, Down, NeuralSpectralBlock2d, Up


def _tiny_network_kwargs() -> dict:
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
        "width": 8,
        "num_token": 4,
        "num_basis": 6,
        "patch_size": 4,
        "n_resblocks": 2,
        "guide_dim": 8,
    }


def test_lso_forward_shape() -> None:
    torch.manual_seed(0)
    model = LSO(**_tiny_network_kwargs())
    out = model(torch.rand(2, 4, 4, 4), torch.rand(2, 2, 16, 16))
    assert out.shape == (2, 4, 16, 16)
    assert torch.isfinite(out).all()


def test_lso_architecture_keeps_upstream_defaults() -> None:
    """The recipe must not have touched upstream's DEFAULT_LSO_ARGS."""
    model = LSO(hsi_channels=128, msi_channels=8, scale=4)

    assert model.width == 64
    assert model.guide_dim == 128
    assert model.n_resblocks == 4
    assert model.patch_size == 4

    assert sum(isinstance(m, NeuralSpectralBlock2d) for m in model.modules()) == 3
    assert sum(isinstance(m, Down) for m in model.modules()) == 2
    assert sum(isinstance(m, Up) for m in model.modules()) == 2
    first = next(m for m in model.modules() if isinstance(m, NeuralSpectralBlock2d))
    assert first.num_token == 8
    assert first.num_basis == 24
    assert first.head == 8


def test_lso_up_concatenates_the_msi_band_count_not_three() -> None:
    """``Up`` hardcodes ``in_channels + 3`` upstream; here it is the MSI width."""
    model = LSO(**_tiny_network_kwargs())
    # Up(width + width, width, msi_channels) -> DoubleConv(in + msi, width, ...)
    expected = (model.width + model.width) + 2
    assert model.up1.conv.double_conv[0].in_channels == expected
    assert model.up2.conv.double_conv[0].in_channels == expected

    wide = LSO(**{**_tiny_network_kwargs(), "msi_channels": 8})
    assert wide.up1.conv.double_conv[0].in_channels == (wide.width * 2) + 8


def test_lso_spatial_encoder_sees_both_observations() -> None:
    model = LSO(**{**_tiny_network_kwargs(), "hsi_channels": 128, "msi_channels": 8})
    assert model.spatial_encoder.head[0].in_channels == 128 + 8


def test_lso_keeps_the_upstream_state_dict_layout() -> None:
    """The port must stay strict-loadable from the released LSO checkpoints.

    All four of them (CAVE and Harvard, x4 and x8) carry 69 tensors.  The port
    reproduces that exact key set, including the four ``sub_mean``/``add_mean``
    tensors that ``EDSR.forward`` never reads but upstream's own
    ``load_state_dict`` would refuse to see dropped.
    """
    model = LSO(hsi_channels=31, msi_channels=3, scale=4)
    state = model.state_dict()
    assert len(state) == 69

    for name in ("sub_mean", "add_mean"):
        weight = state[f"spatial_encoder.{name}.weight"]
        bias = state[f"spatial_encoder.{name}.bias"]
        assert weight.shape == (3, 3, 1, 1)
        assert bias.shape == (3,)
        assert not model.spatial_encoder.get_parameter(f"{name}.weight").requires_grad


def test_lso_mean_shift_tensors_are_unreachable_from_the_forward_pass() -> None:
    torch.manual_seed(0)
    model = LSO(**_tiny_network_kwargs()).eval()
    lr_hsi = torch.rand(1, 4, 4, 4)
    hr_msi = torch.rand(1, 2, 16, 16)
    with torch.no_grad():
        baseline = model(lr_hsi, hr_msi)
        for name in ("sub_mean", "add_mean"):
            layer = getattr(model.spatial_encoder, name)
            layer.weight.add_(1.0)
            layer.bias.add_(1.0)
        perturbed = model(lr_hsi, hr_msi)
    assert torch.equal(baseline, perturbed)


def test_lso_scale_is_configurable() -> None:
    model = LSO(**{**_tiny_network_kwargs(), "scale": 8})
    out = model(torch.rand(1, 4, 2, 2), torch.rand(1, 2, 16, 16))
    assert out.shape == (1, 4, 16, 16)


def test_lso_handles_feature_maps_that_patch_size_does_not_tile() -> None:
    """The ladder halves twice, so ``process2``/``process3`` can see odd sizes."""
    torch.manual_seed(0)
    model = LSO(**_tiny_network_kwargs())
    # 12 -> 6 -> 3, and 3 is not a multiple of the patch size of 4.
    out = model(torch.rand(1, 4, 3, 3), torch.rand(1, 2, 12, 12))
    assert out.shape == (1, 4, 12, 12)

    block = NeuralSpectralBlock2d(8, 6, (4, 4), 4)
    assert block(torch.rand(1, 8, 3, 3)).shape == (1, 8, 3, 3)


def test_lso_predicts_a_residual_over_the_upsampled_lr_hsi() -> None:
    torch.manual_seed(0)
    model = LSO(**_tiny_network_kwargs())
    with torch.no_grad():
        model.outc.conv[2].weight.zero_()
        model.outc.conv[2].bias.zero_()
        lr_hsi = torch.rand(1, 4, 4, 4)
        out = model(lr_hsi, torch.rand(1, 2, 16, 16))
    baseline = F.interpolate(lr_hsi, scale_factor=4, mode="bicubic", align_corners=False)
    assert torch.allclose(out, baseline, atol=1e-6)


def test_lso_rejects_inputs_that_miss_the_ladder() -> None:
    model = LSO(**_tiny_network_kwargs()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should carry"):
        model(torch.rand(1, 5, 4, 4), torch.rand(1, 2, 16, 16))
    with torch.no_grad(), pytest.raises(ValueError, match="HR MSI should carry"):
        model(torch.rand(1, 4, 4, 4), torch.rand(1, 3, 16, 16))
    with torch.no_grad(), pytest.raises(ValueError, match="divisible by 4"):
        model(torch.rand(1, 4, 4, 4), torch.rand(1, 2, 15, 16))
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should be"):
        model(torch.rand(1, 4, 3, 3), torch.rand(1, 2, 16, 16))


def test_lso_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = LSOLightningModule(
        model_config={
            "name": "LSO",
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
        "lr_hsi": torch.rand(2, 4, 4, 4),
        "hr_msi": torch.rand(2, 2, 16, 16),
        "gt_hsi": torch.rand(2, 4, 16, 16),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_lso_recipe_composes() -> None:
    from hydra import compose, initialize_config_dir
    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/lso_x4"])

    assert config.experiment.name == "lso_chikusei_x4"
    assert config.model.name == "LSO"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200
    assert list(config.training.loss.regularization) == []

    model = _build_model(config)
    assert type(model).__name__ == "LSOLightningModule"
    assert type(model.network).__name__ == "LSO"
