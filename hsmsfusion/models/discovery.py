"""Explicit one-folder-per-model discovery with optional dependency reporting."""

from __future__ import annotations

import importlib
import pkgutil

from hsmsfusion.core.registry import MODEL_REGISTRY


def discover_models() -> None:
    """Import each model package and retain missing-dependency diagnostics."""
    import hsmsfusion.models as models_package

    for module in pkgutil.iter_modules(models_package.__path__):
        if not module.ispkg or module.name.startswith("_"):
            continue
        package_name = module.name
        existing = {status.name for status in MODEL_REGISTRY.statuses()}
        if package_name in existing or package_name in MODEL_REGISTRY.names():
            continue
        try:
            importlib.import_module(f"{models_package.__name__}.{package_name}")
        except ModuleNotFoundError as error:
            MODEL_REGISTRY.mark_unavailable(package_name, f"missing dependency: {error.name}")
        except Exception as error:  # pragma: no cover - defensive discovery boundary
            MODEL_REGISTRY.mark_broken(package_name, f"{type(error).__name__}: {error}")

