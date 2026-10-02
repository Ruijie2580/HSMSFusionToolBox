"""Dataset package discovery."""

from __future__ import annotations

import importlib
import pkgutil

from hsmsfusion.core.registry import DATASET_REGISTRY


def discover_datasets() -> None:
    """Import dataset packages once so their explicit decorators run."""
    import hsmsfusion.data.datasets as datasets_package

    for module in pkgutil.iter_modules(datasets_package.__path__):
        if not module.ispkg:
            continue
        name = module.name
        if name in DATASET_REGISTRY.names():
            continue
        importlib.import_module(f"{datasets_package.__name__}.{name}")

