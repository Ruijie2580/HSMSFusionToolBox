"""Shared utilities for HSMSFusion toolbox visualization scripts.

This module extracts common HDF5 I/O, output discovery, model name resolution,
and matplotlib setup used by ``plot_residual_comparison.py``,
``plot_band_psnr.py``, and ``visualize_tensorboard_runs.py``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib shared setup
# ---------------------------------------------------------------------------

_FIGURE_FONT = "Times New Roman"


def setup_matplotlib_font() -> None:
    """Configure Matplotlib to use Times New Roman with STIX math text."""

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": _FIGURE_FONT, "mathtext.fontset": "stix"})


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------


def registered_model_display_names() -> dict[str, str]:
    """Return ``{casefolded_name: canonical_name}`` from the model registry.

    Returns an empty dict when the toolbox package is not importable.
    """

    try:
        from hsmsfusion.core.registry import MODEL_REGISTRY
        from hsmsfusion.models import discover_models

        discover_models()
        return {status.name.casefold(): status.name for status in MODEL_REGISTRY.statuses()}
    except Exception:  # pragma: no cover - optional third-party model imports
        return {}


# ---------------------------------------------------------------------------
# Output prediction discovery
# ---------------------------------------------------------------------------


def discover_output_predictions(
    output_root: Path,
) -> dict[str, list[tuple[str, Path]]]:
    """Discover ``predictions.h5`` files under the toolbox output layout.

    Returns ``{benchmark_name: [(display_name, predictions_path), ...]}``.
    """

    output_root = output_root.expanduser()
    if not output_root.is_dir():
        return {}
    display_names = registered_model_display_names()
    candidates: dict[str, list[tuple[str, Path]]] = {}
    for benchmark_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        methods = [
            (
                display_names.get(method_dir.name.casefold(), method_dir.name),
                method_dir / "predictions.h5",
            )
            for method_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir())
            if (method_dir / "predictions.h5").is_file()
        ]
        if methods:
            candidates[benchmark_dir.name] = methods
    return candidates


# ---------------------------------------------------------------------------
# Dataset path inference
# ---------------------------------------------------------------------------


def infer_target_path(dataset_root: Path, benchmark_name: str) -> Path:
    """Infer ``Dataset/<dataset>/test.h5`` from a benchmark like ``chikusei_x4``."""

    match = re.fullmatch(r"(?P<dataset>.+)_x\d+", benchmark_name, flags=re.IGNORECASE)
    dataset_name = match.group("dataset") if match else benchmark_name
    dataset_root = dataset_root.expanduser()
    if dataset_root.is_dir():
        for child in dataset_root.iterdir():
            if child.is_dir() and child.name.casefold() == dataset_name.casefold():
                return child / "test.h5"
    return dataset_root / dataset_name / "test.h5"


# ---------------------------------------------------------------------------
# HDF5 reading and validation
# ---------------------------------------------------------------------------


def validate_hsi_dataset(
    dataset: h5py.Dataset,
    *,
    path: Path,
    key: str,
) -> tuple[int, int, int, int]:
    """Validate that an HDF5 dataset is NCHW float and return its shape."""

    if dataset.ndim != 4:
        raise ValueError(f"{path}:{key} must be NCHW, got {dataset.shape}.")
    if not np.issubdtype(dataset.dtype, np.floating):
        raise ValueError(f"{path}:{key} must use a floating dtype, got {dataset.dtype}.")
    return tuple(int(v) for v in dataset.shape)


def read_hsi_sample(path: Path, dataset_key: str, sample_index: int) -> np.ndarray:
    """Read one NCHW sample from an HDF5 file as float32 CHW."""

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"HSI file does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{path} does not contain dataset {dataset_key!r}.")
        dataset = handle[dataset_key]
        shape = validate_hsi_dataset(dataset, path=path, key=dataset_key)
        if not 0 <= sample_index < shape[0]:
            raise IndexError(
                f"sample_index={sample_index} is outside [0, {shape[0] - 1}] for {path}."
            )
        sample = np.asarray(dataset[sample_index], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError(f"{path}:{dataset_key}[{sample_index}] contains NaN or Inf.")
    return sample


def sample_count(path: Path, dataset_key: str = "HRHS") -> int:
    """Return the number of samples in an NCHW HDF5 dataset."""

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"HSI file does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{path} does not contain dataset {dataset_key!r}.")
        shape = validate_hsi_dataset(handle[dataset_key], path=path, key=dataset_key)
        return shape[0]


# ---------------------------------------------------------------------------
# Cached target loader (avoids re-reading shared target per method)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedTarget:
    """In-memory copy of the full NCHW target cube for one benchmark."""

    data: np.ndarray  # shape (N, C, H, W), float32
    path: Path

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def band_count(self) -> int:
        return int(self.data.shape[1])


def load_target_cube(
    target_path: Path,
    *,
    dataset_key: str = "HRHS",
    image_indices: Sequence[int] | None = None,
) -> CachedTarget:
    """Load the full target cube (or a subset of samples) into memory once."""

    target_path = target_path.expanduser()
    if not target_path.is_file():
        raise FileNotFoundError(f"Target HSI file does not exist: {target_path}")
    with h5py.File(target_path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{target_path} does not contain dataset {dataset_key!r}.")
        dataset = handle[dataset_key]
        shape = validate_hsi_dataset(dataset, path=target_path, key=dataset_key)
        if image_indices is None:
            data = np.asarray(dataset[:], dtype=np.float32)
        else:
            indices = sorted(set(image_indices))
            invalid = [i for i in indices if not 0 <= i < shape[0]]
            if invalid:
                raise IndexError(
                    f"image index {invalid[0]} is outside [0, {shape[0] - 1}] "
                    f"for {target_path}."
                )
            data = np.asarray(dataset[indices], dtype=np.float32)
    return CachedTarget(data=data, path=target_path)


# ---------------------------------------------------------------------------
# Prediction selection helpers
# ---------------------------------------------------------------------------


def select_predictions(
    candidates: Sequence[tuple[str, Path]],
    requested_methods: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    """Select methods case-insensitively while retaining stable output order."""

    if requested_methods is None:
        return list(candidates)
    by_name = {name.casefold(): (name, path) for name, path in candidates}
    selected: list[tuple[str, Path]] = []
    missing: list[str] = []
    for requested in requested_methods:
        candidate = by_name.get(requested.casefold())
        if candidate is None:
            missing.append(requested)
        elif candidate not in selected:
            selected.append(candidate)
    if missing:
        available = ", ".join(name for name, _ in candidates)
        raise ValueError(f"Unknown method(s): {', '.join(missing)}. Available: {available}.")
    return selected


def resolve_benchmark(
    discovered: dict[str, list[tuple[str, Path]]],
    benchmark: str | None,
) -> tuple[str, list[tuple[str, Path]]]:
    """Resolve a benchmark name from discovered outputs."""

    if benchmark is None:
        if len(discovered) == 1:
            return next(iter(discovered.items()))
        available = ", ".join(discovered) or "none"
        raise ValueError(
            f"Select a dataset-scale output folder. Available datasets: {available}."
        )
    for name, predictions in discovered.items():
        if name.casefold() == benchmark.casefold():
            return name, predictions
    available = ", ".join(discovered) or "none"
    raise ValueError(
        f"No exported predictions found for dataset {benchmark!r}. Available: {available}."
    )
