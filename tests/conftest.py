"""Shared pytest helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "configs" / "experiment"


@pytest.fixture
def require_recipe():
    """Skip a test whose experiment recipe is not published in this checkout.

    Only the faithfully reproduced recipes ship with the repository; the rest
    stay in the local working tree, so a published clone must skip rather than
    fail when one of them is absent.
    """

    def _require(relative: str) -> None:
        if not (_EXPERIMENT_ROOT / f"{relative}.yaml").is_file():
            pytest.skip(f"experiment recipe '{relative}' is not published here")

    return _require
