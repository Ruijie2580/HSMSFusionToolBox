"""Smoke tests for the S3O port."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hsmsfusion.models.S3O import network as s3o_network
from hsmsfusion.models.S3O.lightning_module import S3OLightningModule
from hsmsfusion.models.S3O.network import (
    S3O,
    Block,
    ImplicitHead,
    S3OBackbone,
    Upsampler,
)

# mamba_ssm's selective_scan_fn is a CUDA kernel and the port deliberately does
# not fall back to the slower but numerically different selective_scan_ref.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="S3O's state-space scans need mamba_ssm's CUDA kernel",
)


def _tiny_network_kwargs() -> dict:
    return {
        "hsi_channels": 4,
        "msi_channels": 2,
        "scale": 4,
        "n_hidden": 8,
        "guide_dim": 8,
        "depths": [1, 1, 1],
        "drop_path_rate": 0.0,
    }


def _tiny_model() -> S3O:
    return S3O(**_tiny_network_kwargs())


def _blocks(model: S3O) -> list[Block]:
    return [m for m in model.modules() if isinstance(m, Block)]


def _drop_prob(block: Block) -> float:
    """``Block`` uses ``nn.Identity`` when its drop path rate is zero."""
    return getattr(block.drop_path, "drop_prob", 0.0)


def _lightning_module(**extra) -> S3OLightningModule:
    model_config = {"name": "S3O", "params": _tiny_network_kwargs()}
    model_config.update(extra)
    return S3OLightningModule(
        model_config=model_config,
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


@requires_cuda
def test_s3o_forward_shape() -> None:
    torch.manual_seed(0)
    model = _tiny_model().cuda().eval()
    with torch.no_grad():
        out = model(
            torch.rand(2, 4, 4, 4, device="cuda"),
            torch.rand(2, 2, 16, 16, device="cuda"),
        )
    assert out.shape == (2, 4, 16, 16)
    assert torch.isfinite(out).all()


def test_s3o_architecture_keeps_upstream_defaults() -> None:
    """The recipe must not have touched upstream's Chikusei wrapper defaults."""
    model = S3O(hsi_channels=128, msi_channels=8, scale=4)

    assert model.n_hidden == 64
    # guide_dim 96 is the Chikusei/Harvard value; CAVE upstream uses 128.
    assert model.spa1.scale1[0].in_channels == 128 + 8
    assert model.spa1.out[0].out_channels == 96
    assert model.preprocess.linear_pre[0].in_features == 96
    assert model.preprocess.linear_pre[0].out_features == 2 * 64

    backbone = model.blocks.ssm
    assert isinstance(backbone, S3OBackbone)
    assert backbone.dim == 64
    # Three stages doubling the width, then a 1x1 adapter back to 64.
    assert backbone.channel_adapter.in_channels == 4 * 64
    assert backbone.channel_adapter.out_channels == 64
    assert len(backbone.stages) == 3
    assert isinstance(backbone.upsampler, Upsampler)
    assert sum(isinstance(m, torch.nn.PixelShuffle) for m in backbone.upsampler) == 2

    block = _blocks(model)[0]
    assert block.logA_s.shape == (64, 32)  # N_state_s
    assert block.logA_c.shape == (64, 32)  # N_state_c
    assert block.A_margin == 1e-3
    assert block.sel_groups == 4
    assert block.sel_kernel == 5
    assert block.sel_ratio == 0.10
    assert isinstance(block.enhance, torch.nn.Sequential)

    assert model.recon.proj.out_channels == int(64 * 1.5)
    assert model.recon.pw.out_channels == 128


def test_s3o_drop_path_follows_upstream() -> None:
    """``S3OEncoderBlock`` hardcodes 0.1 upstream; the deepest block keeps it."""
    model = S3O(hsi_channels=4, msi_channels=2, scale=4, n_hidden=8, guide_dim=8)
    rates = [_drop_prob(block) for block in _blocks(model)]
    # ``torch.linspace`` hands back float32-derived Python floats, hence approx.
    assert rates == pytest.approx([0.0, 0.05, 0.1])

    deterministic = _tiny_model()
    assert all(
        isinstance(block.drop_path, torch.nn.Identity) for block in _blocks(deterministic)
    )


def test_s3o_single_stage_skips_the_upsampler() -> None:
    model = S3O(hsi_channels=4, msi_channels=2, scale=4, n_hidden=8, guide_dim=8, depths=[1])
    assert isinstance(model.blocks.ssm.upsampler, torch.nn.Identity)
    assert len(model.blocks.ssm.stages) == 1


def test_s3o_spatial_pyramid_sees_both_observations() -> None:
    """The pyramid's first conv is the HSI-MSI concatenation, in upstream's order."""
    model = S3O(hsi_channels=31, msi_channels=3, scale=4)
    assert model.spa1.scale1[0].in_channels == 34
    # The stride-2 and stride-4 branches pair with bilinear x2 / x4.
    assert model.spa1.scale1_2[0].stride == (2, 2)
    assert model.spa1.scale1_4[0].stride == (4, 4)


def test_s3o_selects_the_configured_token_fraction() -> None:
    """The enhancement budget is ``sel_ratio`` of the tokens, at least one."""
    model = S3O(
        hsi_channels=4, msi_channels=2, scale=4, n_hidden=8, guide_dim=8, depths=[1], sel_ratio=0.25
    )
    block = _blocks(model)[0]
    assert block.sel_ratio == 0.25

    u_s = torch.rand(2, 64, 8)
    u_c = torch.rand(2, 64, 8)
    assert block._select_topk_indices(u_s, u_c, 8, 8).shape == (2, 16)

    block.sel_ratio = 0.0
    # ``max(1, ...)`` keeps a degenerate ratio from selecting nothing.
    assert block._select_topk_indices(u_s, u_c, 8, 8).shape == (2, 1)


def test_s3o_set_enhancement_ratio_reaches_every_block() -> None:
    model = _tiny_model()
    model.set_enhancement_ratio(0.0)
    assert [block.sel_ratio for block in _blocks(model)] == [0.0, 0.0, 0.0]
    model.set_enhancement_ratio(0.075)
    assert [block.sel_ratio for block in _blocks(model)] == [0.075, 0.075, 0.075]


def test_s3o_enhancement_curriculum_matches_upstream() -> None:
    """Upstream ``--sel_warmup 0.10 --sel_ramp 0.20 --sel_max 0.10`` over 200 epochs."""
    model = _lightning_module()
    assert model.sel_warmup_fraction == 0.10
    assert model.sel_ramp_fraction == 0.20
    assert model.sel_max_ratio == 0.10

    ratio = model.enhancement_ratio_for_epoch
    assert ratio(0, 200) == 0.0
    assert ratio(19, 200) == 0.0  # warmup is 20 epochs
    assert ratio(20, 200) == 0.0
    assert ratio(40, 200) == pytest.approx(0.05)  # halfway through the 40-epoch ramp
    assert ratio(59, 200) == pytest.approx(0.0975)
    assert ratio(60, 200) == pytest.approx(0.10)
    assert ratio(199, 200) == pytest.approx(0.10)


def test_s3o_blocks_start_at_the_plateau_ratio() -> None:
    """A checkpoint evaluated without training keeps the test-time ratio."""
    model = _lightning_module(progressive_enhancement={"max_ratio": 0.4})
    assert model.sel_max_ratio == 0.4
    assert [block.sel_ratio for block in _blocks(model.network)] == [0.4, 0.4, 0.4]


def test_s3o_epoch_hook_applies_the_curriculum() -> None:
    model = _lightning_module()
    model._trainer = SimpleNamespace(current_epoch=40, max_epochs=200)
    logged: dict[str, float] = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

    model.on_train_epoch_start()

    assert logged["train/sel_ratio"] == pytest.approx(0.05)
    assert [block.sel_ratio for block in _blocks(model.network)] == [0.05, 0.05, 0.05]


@requires_cuda
def test_s3o_runs_at_any_resolution_from_one_instance() -> None:
    """Upstream freezes H/W per instance and rebuilds at test time; this port does not.

    The toolbox keeps one instance across 64x64 training patches and 256x256 test
    scenes, so the spatial size must come from the input -- including for
    non-square maps, where upstream's ``int(N ** 0.5)`` would silently mis-reshape.
    """
    model = _tiny_model().cuda().eval()
    shapes_before = {name: tensor.shape for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        small = model(
            torch.rand(1, 4, 4, 4, device="cuda"), torch.rand(1, 2, 16, 16, device="cuda")
        )
        wide = model(
            torch.rand(1, 4, 16, 32, device="cuda"), torch.rand(1, 2, 64, 128, device="cuda")
        )

    assert small.shape == (1, 4, 16, 16)
    assert wide.shape == (1, 4, 64, 128)
    assert torch.isfinite(wide).all()
    # No parameter or buffer may encode the spatial size.
    assert {name: tensor.shape for name, tensor in model.state_dict().items()} == shapes_before


@requires_cuda
def test_s3o_predicts_a_residual_over_the_upsampled_lr_hsi() -> None:
    torch.manual_seed(0)
    model = _tiny_model().cuda().eval()
    lr_hsi = torch.rand(1, 4, 4, 4, device="cuda")
    with torch.no_grad():
        model.recon.pw.weight.zero_()
        model.recon.pw.bias.zero_()
        out = model(lr_hsi, torch.rand(1, 2, 16, 16, device="cuda"))
    baseline = F.interpolate(lr_hsi, scale_factor=4, mode="bicubic", align_corners=False)
    assert torch.allclose(out, baseline, atol=1e-6)


def test_s3o_rejects_inputs_that_miss_the_ladder() -> None:
    """Geometry is validated before any scan, so this holds without CUDA."""
    model = _tiny_model().eval()
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should carry"):
        model(torch.rand(1, 5, 4, 4), torch.rand(1, 2, 16, 16))
    with torch.no_grad(), pytest.raises(ValueError, match="HR MSI should carry"):
        model(torch.rand(1, 4, 4, 4), torch.rand(1, 3, 16, 16))
    with torch.no_grad(), pytest.raises(ValueError, match="divisible by 4"):
        model(torch.rand(1, 4, 4, 4), torch.rand(1, 2, 18, 18))
    with torch.no_grad(), pytest.raises(ValueError, match="LR HSI should be"):
        model(torch.rand(1, 4, 3, 3), torch.rand(1, 2, 16, 16))


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="the guard only fires on a CPU-only run"
)
def test_s3o_explains_the_cuda_requirement() -> None:
    model = _tiny_model().eval()
    with torch.no_grad(), pytest.raises(RuntimeError, match="selective_scan_fn"):
        model(torch.rand(1, 4, 4, 4), torch.rand(1, 2, 16, 16))


def test_s3o_drops_the_upstream_dead_code() -> None:
    """``DSDownsample``, the coordinate plumbing and ``spec_dim`` are gone."""
    assert not hasattr(s3o_network, "DSDownsample")

    head = inspect.signature(ImplicitHead.forward).parameters
    assert list(head) == ["self", "feat_hw"]

    head_init = inspect.signature(ImplicitHead.__init__).parameters
    assert "coord_dim" not in head_init
    assert "n_layers" not in head_init

    encoder = inspect.signature(s3o_network.S3OEncoderBlock.__init__).parameters
    assert "spec_dim" not in encoder

    # The spatial size is an input, not a constructor argument.
    block = inspect.signature(Block.__init__).parameters
    assert "H" not in block
    assert "W" not in block

    model = _tiny_model()
    assert not hasattr(model, "H")
    assert not hasattr(_blocks(model)[0], "H")


@requires_cuda
def test_s3o_lightning_training_step() -> None:
    torch.manual_seed(0)
    model = _lightning_module().cuda()
    batch = {
        "lr_hsi": torch.rand(2, 4, 4, 4, device="cuda"),
        "hr_msi": torch.rand(2, 2, 16, 16, device="cuda"),
        "gt_hsi": torch.rand(2, 4, 16, 16, device="cuda"),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_s3o_recipe_composes() -> None:
    from hydra import compose, initialize_config_dir
    from hsmsfusion.cli import _build_model

    config_directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config", overrides=["experiment=chikusei/s3o_x4"])

    assert config.experiment.name == "s3o_chikusei_x4"
    assert config.model.name == "S3O"
    assert config.model.params.hsi_channels == 128
    assert config.model.params.msi_channels == 8
    assert config.model.params.scale == 4
    assert config.model.params.n_hidden == 64
    assert config.model.params.guide_dim == 96
    assert config.model.progressive_enhancement.max_ratio == 0.10
    assert config.model.inference.mode == "full"
    assert config.training.optimizer.lr == 1e-4
    assert config.training.trainer.max_epochs == 200
    assert list(config.training.loss.regularization) == []

    model = _build_model(config)
    assert type(model).__name__ == "S3OLightningModule"
    assert type(model.network).__name__ == "S3O"
