"""A vehicle's initial orbit, as classical elements."""

from dataclasses import dataclass

import numpy as np
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities.macros import D2R

__all__ = ["OrbitElements"]


@dataclass(frozen=True, kw_only=True)
class OrbitElements:
    """Classical orbital elements of the initial orbit, about the central body.

    Attributes:
        semi_major_axis_m: Semi-major axis, in metres.
        eccentricity: Eccentricity, dimensionless, `0 <= e < 1`.
        inclination_rad: Inclination, in radians.
        raan_rad: Right ascension of the ascending node, in radians.
        arg_periapsis_rad: Argument of periapsis, in radians.
        true_anomaly_rad: True anomaly at t = 0, in radians.

    Raises:
        ValueError: `semi_major_axis_m` is not positive, or `eccentricity` is outside [0, 1).
    """

    semi_major_axis_m: float = 6871.0e3
    eccentricity: float = 0.0001
    inclination_rad: float = 45.0 * D2R
    raan_rad: float = 48.2 * D2R
    arg_periapsis_rad: float = 347.8 * D2R
    true_anomaly_rad: float = 85.3 * D2R

    def __post_init__(self) -> None:
        if not self.semi_major_axis_m > 0:
            raise ValueError(f"semi-major axis must be positive, got {self.semi_major_axis_m} m")
        if not 0 <= self.eccentricity < 1:
            raise ValueError(f"eccentricity must be in [0, 1), got {self.eccentricity}")

    def to_inertial(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity of the orbit at its true anomaly.

        Args:
            mu: The central body's gravitational parameter, in m^3/s^2.

        Returns:
            `(r_N, v_N)`: inertial position in metres and velocity in m/s, each shape `(3,)`.
        """
        elements = orbitalMotion.ClassicElements()
        elements.a = self.semi_major_axis_m
        elements.e = self.eccentricity
        elements.i = self.inclination_rad
        elements.Omega = self.raan_rad
        elements.omega = self.arg_periapsis_rad
        elements.f = self.true_anomaly_rad
        r_N, v_N = orbitalMotion.elem2rv(mu, elements)
        return np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float)
