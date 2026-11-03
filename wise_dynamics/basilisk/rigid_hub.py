"""A spacecraft's rigid hub: its mass and inertia."""

from dataclasses import dataclass

__all__ = ["RigidHub"]


@dataclass(frozen=True, kw_only=True)
class RigidHub:
    """The spacecraft's rigid hub.

    Attributes:
        mass_kg: Mass, in kilograms.
        inertia_B_kg_m2: Inertia tensor about the centre of mass in body axes, in kg*m^2, as
            three rows.
    """

    mass_kg: float = 750.0
    inertia_B_kg_m2: tuple[tuple[float, float, float], ...] = (
        (900.0, 0.0, 0.0),
        (0.0, 800.0, 0.0),
        (0.0, 0.0, 600.0),
    )
