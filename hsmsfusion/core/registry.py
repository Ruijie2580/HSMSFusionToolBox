"""Small explicit registries used by toolbox extension points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from hsmsfusion.core.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
)

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryStatus:
    """Availability information for a registered or discovered extension."""

    name: str
    status: str
    reason: str | None = None


class Registry(Generic[T]):
    """Name-to-object registry with explicit unavailable and broken states."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._entries: dict[str, T] = {}
        self._status: dict[str, RegistryStatus] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def decorator(entry: T) -> T:
            entry_name = name or getattr(entry, "registry_name", None) or getattr(
                entry, "__name__", None
            )
            if not entry_name:
                raise ConfigurationError(
                    f"Cannot infer a name for {self.category} registry entry."
                )
            if entry_name in self._entries or entry_name in self._status:
                raise ConfigurationError(
                    f"Duplicate {self.category} registry name: {entry_name!r}."
                )
            self._entries[entry_name] = entry
            self._status[entry_name] = RegistryStatus(entry_name, "available")
            return entry

        return decorator

    def mark_unavailable(self, name: str, reason: str) -> None:
        if name in self._entries:
            return
        self._status[name] = RegistryStatus(name, "unavailable", reason)

    def mark_broken(self, name: str, reason: str) -> None:
        if name in self._entries:
            return
        self._status[name] = RegistryStatus(name, "broken", reason)

    def get(self, name: str) -> T:
        if name in self._entries:
            return self._entries[name]
        status = self._status.get(name)
        if status and status.status == "unavailable":
            raise DependencyUnavailableError(
                f"{self.category} {name!r} is unavailable: {status.reason}"
            )
        if status and status.status == "broken":
            raise ConfigurationError(
                f"{self.category} {name!r} is broken: {status.reason}"
            )
        available = ", ".join(sorted(self._entries)) or "<none>"
        raise ConfigurationError(
            f"Unknown {self.category} {name!r}. Available: {available}."
        )

    def statuses(self) -> list[RegistryStatus]:
        return [self._status[name] for name in sorted(self._status)]

    def names(self) -> list[str]:
        return sorted(self._entries)


MODEL_REGISTRY: Registry[Any] = Registry("model")
DATASET_REGISTRY: Registry[Any] = Registry("dataset")
SPECTRAL_OPERATOR_REGISTRY: Registry[Any] = Registry("spectral operator")
SPATIAL_OPERATOR_REGISTRY: Registry[Any] = Registry("spatial operator")
METRIC_REGISTRY: Registry[Any] = Registry("metric")
LOSS_REGISTRY: Registry[Any] = Registry("loss")
