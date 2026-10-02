"""Registry integrity contracts: discovery must populate every registry.

Registration happens through import side effects (see docs/architecture.md), so
a deleted import can silently empty a registry. These tests pin the exact
expected contents after the discovery/import entry points run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from hsmsfusion.core.registry import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    SPATIAL_OPERATOR_REGISTRY,
    SPECTRAL_OPERATOR_REGISTRY,
)
from hsmsfusion.data.discovery import discover_datasets
from hsmsfusion.models.discovery import discover_models

_EXPECTED_DATASETS = {
    "cave",
    "chikusei",
    "harvard",
    "houston",
    "icvl",
    "paviac",
    "wdcmall",
    "xiongan",
}
_EXPECTED_SPECTRAL_OPERATORS = {"gaussian_srf", "tabulated_srf"}
_EXPECTED_SPATIAL_OPERATORS = {"gaussian_decimation"}
_EXPECTED_METRICS = {
    "psnr_global",
    "rmse_global",
    "ssim_band_mean",
    "sam_rad",
    "sam_deg",
    "ergas",
    "uiqi",
}
_EXPECTED_LOSSES = {
    "l1",
    "l2",
    "char",
    "cycle_consistency",
    "sam",
    "ssim",
    "tv",
    "spatial_edge",
    "spectral_edge",
}


def test_dataset_registry_discovers_exactly_the_eight_benchmark_datasets() -> None:
    discover_datasets()

    assert set(DATASET_REGISTRY.names()) == _EXPECTED_DATASETS


def test_operator_registries_after_degradations_import() -> None:
    import hsmsfusion.degradations  # noqa: F401  (import for registry side effects)

    assert _EXPECTED_SPECTRAL_OPERATORS <= set(SPECTRAL_OPERATOR_REGISTRY.names())
    assert _EXPECTED_SPATIAL_OPERATORS <= set(SPATIAL_OPERATOR_REGISTRY.names())


def test_metric_registry_after_metrics_import() -> None:
    import hsmsfusion.metrics  # noqa: F401  (import for registry side effects)

    assert _EXPECTED_METRICS <= set(METRIC_REGISTRY.names())


def test_loss_registry_after_losses_import() -> None:
    import hsmsfusion.losses  # noqa: F401  (import for registry side effects)

    assert _EXPECTED_LOSSES <= set(LOSS_REGISTRY.names())


def _tracked_model_packages() -> set[str]:
    """Derive the published model package names from git's tracked file list."""
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "hsmsfusion/models/"],
        cwd=str(repository_root),
        capture_output=True,
        text=True,
        check=True,
    )
    packages: set[str] = set()
    for line in result.stdout.splitlines():
        parts = PurePosixPath(line).parts
        if len(parts) > 3:  # hsmsfusion/models/<Package>/<file>
            packages.add(parts[2])
    return packages


def test_model_registry_covers_every_tracked_model_package() -> None:
    """Every published model package must register (local-only packages excluded).

    The tracked package list comes from git, so packages kept local via
    .gitignore (KNLNet, SatFusion, SpecSolver, SSILNet) are never asserted here
    and the test passes on a public clone as well.
    """
    tracked_packages = _tracked_model_packages()
    assert tracked_packages, "git ls-files returned no tracked model packages"

    discover_models()

    missing = tracked_packages - set(MODEL_REGISTRY.names())
    assert not missing, f"tracked model packages missing from MODEL_REGISTRY: {missing}"
