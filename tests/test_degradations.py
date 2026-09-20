from __future__ import annotations

import torch

from hsmsfusion.degradations import build_observation_generator


def _config() -> dict[str, object]:
    return {
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


def test_observation_generator_preserves_contract() -> None:
    generator = build_observation_generator(_config())
    hr_hsi = torch.rand(3, 4, 8, 8)
    observations = generator(hr_hsi)

    assert observations["hr_msi"].shape == (3, 2, 8, 8)
    assert observations["lr_hsi"].shape == (3, 4, 4, 4)
    assert torch.allclose(generator.spectral_operator.srf.sum(dim=1), torch.ones(2))


def test_materialized_observations_equal_batched_forward() -> None:
    generator = build_observation_generator(_config())
    hr_hsi = torch.rand(5, 4, 8, 8)
    direct = generator(hr_hsi)
    materialized = generator.generate_in_batches(hr_hsi, batch_size=2)

    assert torch.allclose(direct["hr_msi"], materialized["hr_msi"])
    assert torch.allclose(direct["lr_hsi"], materialized["lr_hsi"])


def test_tabulated_wv2_srf_is_interpolated_and_normalized() -> None:
    generator = build_observation_generator(
        {
            "spectral": {
                "name": "tabulated_srf",
                "sensor": "wv2",
                "response_file": "WV2_srf.mat",
                "wavelength_start_nm": 363.0,
                "wavelength_end_nm": 1018.0,
                "num_hsi_bands": 128,
                "num_msi_bands": 8,
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
    )
    spectral_operator = generator.spectral_operator
    observations = generator(torch.rand(2, 128, 8, 8))

    assert spectral_operator.response_file == "WV2_srf.mat"
    assert spectral_operator.sensor == "WV2"
    assert spectral_operator.srf.shape == (8, 128)
    assert torch.allclose(spectral_operator.srf.sum(dim=1), torch.ones(8))
    assert observations["hr_msi"].shape == (2, 8, 8, 8)
