"""A plant's standard per-case dispersion, drawn as constructor keyword overrides."""

import inspect
from typing import Any

import numpy as np

__all__ = ["draw_standard"]


def draw_standard(plant_type: type, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw one set of a plant's dispersed constructor keywords.

    For each entry of `plant_type.standard_dispersion`, in its order, the constructor's default
    for that keyword plus three independent zero-mean normal samples with that 1-sigma.

    Args:
        plant_type: A plant class: its `standard_dispersion` maps a 3-vector constructor keyword to
            its per-axis 1-sigma in that keyword's units.
        rng: The generator the samples are drawn from.

    Returns:
        Keyword to its drawn value, a float array of shape `(3,)`, ready for
        `plant_type(**draws)`.

    Raises:
        ValueError: A `standard_dispersion` key is a keyword the constructor lacks or whose
            default is not a 3-vector.
    """
    defaults = inspect.signature(plant_type).parameters
    draws: dict[str, np.ndarray] = {}
    for name, std in plant_type.standard_dispersion.items():
        if name not in defaults:
            raise ValueError(f"{plant_type.__name__} has no keyword {name!r}")
        nominal: Any = np.asarray(defaults[name].default, dtype=float)
        if nominal.shape != (3,):
            raise ValueError(f"{plant_type.__name__}'s {name!r} default is not a 3-vector")
        draws[name] = nominal + rng.normal(0.0, std, size=3)
    return draws
