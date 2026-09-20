"""HDF5 prediction export under the strict one-dataset contract."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from hsmsfusion.core.exceptions import ContractError


def write_prediction_file(path: Path, prediction: np.ndarray) -> None:
    """Atomically write only `/prediction` in float32 NCHW layout."""
    if prediction.ndim != 4:
        raise ContractError(f"prediction must be NCHW, got {prediction.shape}.")
    if prediction.dtype != np.float32:
        raise ContractError(f"prediction must be float32, got {prediction.dtype}.")
    if not np.isfinite(prediction).all():
        raise ContractError("prediction contains NaN or Inf.")
    if np.any(prediction < 0.0) or np.any(prediction > 1.0):
        raise ContractError("prediction must be clipped to [0, 1] before export.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    chunk_shape = (1, prediction.shape[1], prediction.shape[2], prediction.shape[3])
    with h5py.File(temporary_path, "w") as handle:
        handle.create_dataset(
            "prediction",
            data=prediction,
            dtype=np.float32,
            chunks=chunk_shape,
        )
    temporary_path.replace(path)

