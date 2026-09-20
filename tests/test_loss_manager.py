from __future__ import annotations

import pytest
import torch

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.degradations import build_observation_generator
from hsmsfusion.losses import LossManager


def test_loss_manager_composes_weighted_terms_and_backpropagates() -> None:
    prediction = torch.tensor([[[[0.2]], [[0.8]]]], requires_grad=True)
    target = torch.tensor([[[[0.0]], [[1.0]]]])
    manager = LossManager(
        {
            "reconstruction": [
                {"name": "l1", "alias": "reconstruction", "weight": 1.0},
                {"name": "l2", "alias": "squared_error", "weight": 0.5},
            ],
            "regularization": [],
        }
    )

    total, components = manager(prediction, {"gt_hsi": target})

    expected_l1 = torch.tensor(0.2)
    expected_mse = torch.tensor(0.04)
    assert torch.allclose(components["reconstruction"], expected_l1)
    assert torch.allclose(components["squared_error"], expected_mse)
    assert torch.allclose(total, expected_l1 + 0.5 * expected_mse)
    assert torch.allclose(components["total"], total)
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_loss_manager_rejects_legacy_flat_loss_config() -> None:
    with pytest.raises(ConfigurationError, match="Legacy loss configuration"):
        LossManager({"kind": "l1", "weight": 2.0})

    with pytest.raises(ConfigurationError, match="Legacy loss configuration"):
        LossManager({"components": [{"name": "l1", "weight": 1.0}]})


def test_observation_losses_require_known_operators() -> None:
    with pytest.raises(ConfigurationError, match="known degradation operators"):
        LossManager({"reconstruction": [{"name": "cycle_consistency", "weight": 1.0}]})


def test_cycle_consistency_reobserves_known_operators_and_backpropagates() -> None:
    operator_config = {
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
    prediction = torch.rand(1, 4, 8, 8, requires_grad=True)
    observations = build_observation_generator(operator_config)(prediction.detach())
    manager = LossManager(
        {
            "reconstruction": [{"name": "cycle_consistency", "weight": 1.0}],
            "regularization": [],
        },
        operator_config=operator_config,
    )

    total, components = manager(prediction, observations)

    assert torch.allclose(total, torch.zeros_like(total), atol=1e-7)
    assert torch.allclose(components["total"], total)
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_loss_manager_separates_regularization_terms_and_backpropagates() -> None:
    prediction = torch.rand(1, 3, 12, 12, requires_grad=True)
    target = torch.rand_like(prediction)
    manager = LossManager(
        {
            "reconstruction": [{"name": "char", "alias": "char", "weight": 1.0}],
            "regularization": [
                {"name": "sam", "alias": "sam", "weight": 0.1},
                {
                    "name": "ssim",
                    "alias": "ssim",
                    "weight": 0.2,
                    "params": {"window_size": 3},
                },
                {"name": "tv", "alias": "tv", "weight": 0.01},
            ],
        }
    )

    total, components = manager(prediction, {"gt_hsi": target})

    assert set(components) == {
        "char",
        "char_weighted",
        "sam",
        "sam_weighted",
        "ssim",
        "ssim_weighted",
        "tv",
        "tv_weighted",
        "total",
    }
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_gradient_regularizers_support_l1_and_mse_criteria() -> None:
    prediction = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction)
    manager = LossManager(
        {
            "reconstruction": [],
            "regularization": [
                {
                    "name": "spatial_edge",
                    "alias": "spatial_l1",
                    "weight": 1.0,
                    "params": {"criterion": "l1"},
                },
                {
                    "name": "spectral_edge",
                    "alias": "spectral_mse",
                    "weight": 1.0,
                    "params": {"criterion": "mse"},
                },
                {
                    "name": "tv",
                    "alias": "tv_mse",
                    "weight": 0.1,
                    "params": {"criterion": "mse"},
                },
            ],
        }
    )

    total, components = manager(prediction, {"gt_hsi": target})

    assert all(torch.isfinite(value) for value in components.values())
    assert components["spatial_l1"].item() >= 0.0
    assert components["spectral_mse"].item() >= 0.0
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

    with pytest.raises(ConfigurationError, match="criterion"):
        LossManager(
            {
                "regularization": [
                    {"name": "spatial_edge", "params": {"criterion": "char"}}
                ]
            }
        )
