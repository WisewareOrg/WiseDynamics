"""Per-case normal dispersion of a plant's 3-vector configuration fields."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["Dispersion", "apply_draws"]


@dataclass(frozen=True)
class Dispersion:
    """Independent zero-mean normal perturbations of named 3-vector fields of a configuration.

    Attributes:
        std: Configuration field name to its per-axis 1-sigma, in that field's own units. The
            mapping's order is the order values are drawn in, which is part of a session's
            reproducibility.
    """

    std: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "std", MappingProxyType(dict(self.std)))

    def draw(self, nominal: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw one set of dispersed values around a nominal configuration.

        Args:
            nominal: A configuration dataclass instance holding each named field as a 3-vector.
            rng: The generator the perturbations are drawn from, three normal samples per field
                in `std` order.

        Returns:
            Field name to its dispersed value, `nominal`'s value plus the perturbation, as a
            float array of shape `(3,)`.

        Raises:
            ValueError: A name in `std` is a field `nominal` lacks, or holds something other than
                a 3-vector.
        """
        return {
            name: _vector_field(nominal, name) + rng.normal(0.0, std, size=3)
            for name, std in self.std.items()
        }


def apply_draws(config: Any, draws: Mapping[str, Any]) -> Any:
    """Return a copy of a configuration with drawn 3-vector values in place of its own.

    Args:
        config: A frozen configuration dataclass instance.
        draws: Field name to its new 3-vector value.

    Returns:
        A new configuration of the same type, each drawn field replaced by a tuple of floats.

    Raises:
        ValueError: A draw names a field `config` lacks, or a value or field that is not a
            3-vector.
    """
    replacements = {}
    for name, value in draws.items():
        _vector_field(config, name)
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"draw {name!r} has shape {vector.shape}, expected (3,)")
        replacements[name] = tuple(float(component) for component in vector)
    return dataclasses.replace(config, **replacements)


def _vector_field(config: Any, name: str) -> np.ndarray:
    """The named 3-vector field of a configuration dataclass, as a float array."""
    names = {field.name for field in dataclasses.fields(config)}
    if name not in names:
        raise ValueError(
            f"{type(config).__name__} has no field {name!r}; its fields are {sorted(names)}"
        )
    vector = np.asarray(getattr(config, name), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{type(config).__name__}.{name} is not a 3-vector")
    return vector
