"""Experiment-directory and manifest utilities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from hsmsfusion.core.exceptions import ConfigurationError


def as_plain(config: Any) -> Any:
    """Resolve OmegaConf containers into plain Python values."""
    return OmegaConf.to_container(config, resolve=True)


def config_hash(config: DictConfig) -> str:
    """Hash stable experiment content, excluding Hydra and invocation fields."""
    data = as_plain(config)
    for key in ("hydra", "resume", "run_dir", "checkpoint_name", "prediction_path"):
        data.pop(key, None)
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]


def experiment_directory(config: DictConfig) -> Path:
    """Return the unique run directory for the configured experiment name.

    Experiment names are intentionally restricted to one direct child of
    ``runs_dir``. This keeps non-resume replacement safely scoped to
    ``runs/<experiment>/``.
    """

    runs_root = Path(str(config.paths.runs_dir)).resolve()
    experiment_name = str(config.experiment.name).strip()
    if not experiment_name:
        raise ConfigurationError("experiment.name must not be empty.")
    run_directory = (runs_root / experiment_name).resolve()
    if run_directory.parent != runs_root:
        raise ConfigurationError(
            "experiment.name must identify one direct child of paths.runs_dir."
        )
    return run_directory


def benchmark_output_directory(config: DictConfig) -> Path:
    """Return the benchmark output directory grouped by data, scale, and model."""

    output_root = Path(str(config.paths.get("output_dir", "output"))).resolve()
    dataset_name = _output_path_component(str(config.dataset.name), field_name="dataset.name")
    model_name = _output_path_component(str(config.model.name), field_name="model.name")
    scale = int(config.degradation.spatial.scale)
    if scale < 1:
        raise ConfigurationError("degradation.spatial.scale must be a positive integer.")
    return output_root / f"{dataset_name}_x{scale}" / model_name


def _output_path_component(value: str, *, field_name: str) -> str:
    component = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", component):
        raise ConfigurationError(
            f"{field_name} must contain only letters, numbers, underscores, or hyphens "
            "when used to build an output path."
        )
    return component


def create_run_directory(config: DictConfig) -> tuple[Path, bool]:
    """Create a fresh ``runs/<experiment>/`` directory for a non-resume run.

    Any existing record for precisely the same experiment is removed first,
    including checkpoints, logs, predictions, and manifests. The return value
    contains the new directory and whether a previous record was replaced.
    """

    run_directory = experiment_directory(config)
    replaced_existing_record = run_directory.exists()
    if replaced_existing_record:
        if not run_directory.is_dir() or run_directory.is_symlink():
            raise ConfigurationError(
                f"Refusing to replace an unsafe experiment directory: {run_directory}"
            )
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory, replaced_existing_record


def save_resolved_config(config: DictConfig, run_directory: Path) -> None:
    OmegaConf.save(config=config, f=run_directory / "resolved_config.yaml", resolve=True)


def load_resolved_config(run_directory: Path) -> DictConfig:
    path = run_directory / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Run directory has no resolved_config.yaml: {run_directory}")
    return OmegaConf.load(path)


def update_manifest(run_directory: Path, **updates: Any) -> None:
    path = run_directory / "manifest.yaml"
    existing = OmegaConf.load(path) if path.exists() else OmegaConf.create()
    for key, value in updates.items():
        existing[key] = value
    OmegaConf.save(existing, path, resolve=True)
