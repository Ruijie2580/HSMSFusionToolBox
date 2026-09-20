"""Configuration-driven composition of reconstruction and regularization losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import LOSS_REGISTRY
from hsmsfusion.degradations.observation import ObservationGenerator, build_observation_generator
from hsmsfusion.losses.term import ConfigurableLossTerm


@dataclass(frozen=True)
class _Component:
    """One configured loss term and its contribution to the total objective."""

    name: str
    alias: str
    group: str
    weight: float


_LOSS_GROUPS = ("reconstruction", "regularization")


def _component_configs(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Validate and flatten the two explicit loss groups from a model recipe."""

    if "components" in config or "kind" in config or "weight" in config:
        raise ConfigurationError(
            "Legacy loss configuration is unsupported. Use loss.reconstruction and/or "
            "loss.regularization lists with name, alias, weight, and optional params."
        )

    configured_groups = [group for group in _LOSS_GROUPS if group in config]
    if not configured_groups:
        raise ConfigurationError(
            "Loss configuration must define at least one of: loss.reconstruction, "
            "loss.regularization."
        )

    flattened: list[tuple[str, Mapping[str, Any]]] = []
    for group in _LOSS_GROUPS:
        raw_components = config.get(group, [])
        if raw_components is None:
            raw_components = []
        if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
            raise ConfigurationError(f"loss.{group} must be a list of component mappings.")
        for index, component in enumerate(raw_components):
            if not isinstance(component, Mapping):
                raise ConfigurationError(f"loss.{group}[{index}] must be a mapping.")
            if not component.get("name"):
                raise ConfigurationError(f"loss.{group}[{index}].name is required.")
            flattened.append((group, component))

    if not flattened:
        raise ConfigurationError("At least one configured loss component is required.")
    return flattened


class LossManager(nn.Module):
    """Build and combine registry-backed loss terms from a grouped configuration.

    Each configured term contributes ``weight * raw_value`` to ``total``.  Raw
    values and weighted contributions are returned separately for Lightning,
    CSV/TensorBoard logs, and Rich terminal reporting.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        operator_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, Mapping):
            raise ConfigurationError("training.loss must be a mapping.")

        raw_components = _component_configs(config)
        needs_observation_generator = False
        resolved: list[tuple[str, Mapping[str, Any], type[ConfigurableLossTerm]]] = []
        for group, component in raw_components:
            name = str(component["name"])
            try:
                factory = LOSS_REGISTRY.get(name)
            except KeyError as error:
                raise ConfigurationError(str(error)) from error
            if not isinstance(factory, type) or not issubclass(factory, ConfigurableLossTerm):
                raise ConfigurationError(
                    f"Loss registry entry '{name}' must be a ConfigurableLossTerm class."
                )
            category = getattr(factory, "category", None)
            if category != group:
                raise ConfigurationError(
                    f"Loss '{name}' belongs to loss.{category}, not loss.{group}."
                )
            needs_observation_generator |= bool(getattr(factory, "requires_observation_generator", False))
            resolved.append((group, component, factory))

        generator: ObservationGenerator | None = None
        if needs_observation_generator:
            if operator_config is None:
                raise ConfigurationError(
                    "Configured observation-consistency loss requires known degradation operators."
                )
            generator = build_observation_generator(operator_config)

        self.components: list[_Component] = []
        terms: dict[str, ConfigurableLossTerm] = {}
        aliases: set[str] = set()
        for index, (group, component, factory) in enumerate(resolved):
            name = str(component["name"])
            alias = str(component.get("alias") or f"{group}_{name}_{index + 1}")
            if alias in aliases:
                raise ConfigurationError(f"Loss aliases must be unique; duplicate alias '{alias}'.")
            aliases.add(alias)
            try:
                weight = float(component.get("weight", 1.0))
            except (TypeError, ValueError) as error:
                raise ConfigurationError(f"Loss '{alias}' has a non-numeric weight.") from error
            parameters = component.get("params", {})
            if not isinstance(parameters, Mapping):
                raise ConfigurationError(f"loss parameter mapping for '{alias}' must be a mapping.")
            terms[alias] = factory(parameters=parameters, observation_generator=generator)
            self.components.append(_Component(name=name, alias=alias, group=group, weight=weight))
        self.terms = nn.ModuleDict(terms)

    def forward(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = prediction.new_zeros(())
        values: dict[str, torch.Tensor] = {}
        for component in self.components:
            raw_value = self.terms[component.alias](prediction, batch)
            if raw_value.ndim != 0:
                raise ConfigurationError(
                    f"Loss '{component.alias}' must return a scalar, got shape {tuple(raw_value.shape)}."
                )
            weighted_value = raw_value * component.weight
            values[component.alias] = raw_value
            values[f"{component.alias}_weighted"] = weighted_value
            total = total + weighted_value
        values["total"] = total
        return total, values
