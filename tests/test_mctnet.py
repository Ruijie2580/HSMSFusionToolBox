from __future__ import annotations

import torch

from hsmsfusion.models.MCTNet.lightning_module import MCTNetLightningModule


def _module() -> MCTNetLightningModule:
    return MCTNetLightningModule(
        model_config={
            "name": "MCTNet",
            "params": {
                "hsi_channels": 4,
                "msi_channels": 1,
                "scale": 4,
                "base_hr_size": 16,
                "transformer_depth": 1,
                "transformer_heads": 1,
                "transformer1_dim": 4,
                "transformer2_dim": 4,
                "dropout": 0.0,
                "attention_dropout": 0.0,
            },
        },
        optimizer_config={"name": "adam", "lr": 1e-3},
        scheduler_config={"name": "none"},
        loss_config={
            "reconstruction": [
                {"name": "l1", "alias": "fusion_reconstruction", "weight": 1.0},
            ],
            "regularization": [
                {
                    "name": "spatial_edge",
                    "alias": "spatial_edge",
                    "weight": 0.1,
                    "params": {"criterion": "mse"},
                },
                {
                    "name": "spectral_edge",
                    "alias": "spectral_edge",
                    "weight": 0.05,
                    "params": {"criterion": "l1"},
                },
            ],
        },
        metric_config={
            "leaderboard": ["psnr_global", "ssim_band_mean", "sam_deg", "ergas", "rmse_global", "uiqi"],
            "ssim": {"window_size": 3, "sigma": 1.0, "padding": "reflect"},
        },
        scale=4,
    )


def test_mctnet_returns_only_fusion_and_uses_generic_compound_losses() -> None:
    model = _module()
    lr_hsi = torch.rand(2, 4, 4, 4)
    hr_msi = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 4, 16, 16)

    model.train()
    training_prediction = model(lr_hsi, hr_msi)
    assert isinstance(training_prediction, torch.Tensor)
    assert training_prediction.shape == target.shape

    compound_loss, components = model.compute_objective(training_prediction, {"gt_hsi": target})
    assert set(components) == {
        "fusion_reconstruction",
        "fusion_reconstruction_weighted",
        "spatial_edge",
        "spatial_edge_weighted",
        "spectral_edge",
        "spectral_edge_weighted",
        "total",
    }
    assert torch.allclose(
        compound_loss,
        sum(components[f"{component.alias}_weighted"] for component in model.loss_manager.components),
    )

    model.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    loss = model.training_step(
        {"lr_hsi": lr_hsi, "hr_msi": hr_msi, "gt_hsi": target},
        batch_idx=0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        evaluation_prediction = model(lr_hsi[:1], hr_msi[:1])
        whole_image_prediction = model(torch.rand(1, 4, 8, 8), torch.rand(1, 1, 32, 32))
    assert isinstance(evaluation_prediction, torch.Tensor)
    assert evaluation_prediction.shape == (1, 4, 16, 16)
    assert whole_image_prediction.shape == (1, 4, 32, 32)

    model.on_validation_epoch_start()
    validation_output = model.validation_step(
        {"lr_hsi": lr_hsi[:1], "hr_msi": hr_msi[:1], "gt_hsi": target[:1]},
        batch_idx=0,
    )
    assert validation_output["prediction"].shape == (1, 4, 16, 16)

    restored = _module()
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(lr_hsi[:1], hr_msi[:1])
    assert torch.allclose(evaluation_prediction, restored_prediction)
