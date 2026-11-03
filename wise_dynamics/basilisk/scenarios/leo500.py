"""A vehicle in a 500 km, 45 degree low Earth orbit under point-mass Earth gravity.

`wise_dynamics/basilisk/README.md` tabulates its environment and placement with their units and
frames.
"""

from typing import ClassVar

import numpy as np
from Basilisk.utilities import orbitalMotion, simIncludeGravBody
from Basilisk.utilities.macros import D2R

from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["Leo500"]


class Leo500(Scenario):
    """A vehicle tumbling on a classical-element orbit about a point-mass Earth.

    The inertial frame N is Earth-centred, with no rotation and no epoch: point-mass gravity has
    nothing that depends on time. The class attributes define the scenario; `disperse` overrides
    `sigma0_BN` and `omega0_BN_B_rad_s` on one instance.

    Attributes:
        mu_m3_s2: Earth's gravitational parameter, in m^3/s^2, keyed `"earth"`.
        standard_dispersion: The per-axis 1-sigma of `sigma0_BN`, a dimensionless MRP, and of
            `omega0_BN_B_rad_s`, in rad/s.
        semi_major_axis_m: Semi-major axis of the initial orbit, in metres.
        eccentricity: Eccentricity of the initial orbit, `0 <= e < 1`.
        inclination_rad: Inclination, in radians.
        raan_rad: Right ascension of the ascending node, in radians.
        arg_periapsis_rad: Argument of periapsis, in radians.
        true_anomaly_rad: True anomaly at t = 0, in radians.
        sigma0_BN: Initial attitude, the MRP of the body frame relative to N.
        omega0_BN_B_rad_s: Initial body rate relative to N, in body axes, in rad/s.
        step_s: The integration and sensor step, in seconds.
    """

    mu_m3_s2: ClassVar[dict[str, float]] = {"earth": 3.986004360e14}
    standard_dispersion: ClassVar[dict[str, float]] = {
        "sigma0_BN": 0.01,
        "omega0_BN_B_rad_s": 0.2 * D2R,
    }

    semi_major_axis_m: float = 6871.0e3
    eccentricity: float = 0.0001
    inclination_rad: float = 45.0 * D2R
    raan_rad: float = 48.2 * D2R
    arg_periapsis_rad: float = 347.8 * D2R
    true_anomaly_rad: float = 85.3 * D2R
    sigma0_BN: tuple[float, float, float] = (0.1, 0.2, -0.3)
    omega0_BN_B_rad_s: tuple[float, float, float] = (1.0 * D2R, 1.0 * D2R, 1.0 * D2R)
    step_s: float = 0.02

    def initial_state(self) -> dict[str, np.ndarray]:
        """The initial condition the placement gives the vehicle, each a float array of shape
        `(3,)`.

        Returns:
            `sigma0_BN` (MRP, body relative to N), `r0_N` (position of the centre of mass in N,
            m), `v0_N` (its velocity in N, m/s) and `omega0_BN_B` (body rate relative to N in
            body axes, rad/s).

        Raises:
            ValueError: `semi_major_axis_m` is not positive, or `eccentricity` is outside [0, 1).
        """
        r_N, v_N = _orbit_to_inertial(
            self.mu_m3_s2["earth"],
            self.semi_major_axis_m,
            self.eccentricity,
            self.inclination_rad,
            self.raan_rad,
            self.arg_periapsis_rad,
            self.true_anomaly_rad,
        )
        return {
            "sigma0_BN": np.asarray(self.sigma0_BN, dtype=float),
            "r0_N": r_N,
            "v0_N": v_N,
            "omega0_BN_B": np.asarray(self.omega0_BN_B_rad_s, dtype=float),
        }

    def _place(self, vehicle: Vehicle) -> None:
        """Point-mass Earth as the central body, and the initial state on the hub."""
        state = self.initial_state()
        self._gravity = simIncludeGravBody.gravBodyFactory()
        earth = self._gravity.createEarth()
        earth.mu = self.mu_m3_s2["earth"]
        earth.isCentralBody = True
        self._gravity.addBodiesTo(vehicle.spacecraft)

        hub = vehicle.spacecraft.hub
        hub.r_CN_NInit = state["r0_N"]
        hub.v_CN_NInit = state["v0_N"]
        hub.sigma_BNInit = state["sigma0_BN"]
        hub.omega_BN_BInit = state["omega0_BN_B"]


def _orbit_to_inertial(
    mu: float, a_m: float, e: float, i_rad: float, raan_rad: float, argp_rad: float, f_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position (m) and velocity (m/s) of a classical-element orbit.

    Raises:
        ValueError: `a_m` is not positive, or `e` is outside [0, 1).
    """
    if not a_m > 0:
        raise ValueError(f"semi-major axis must be positive, got {a_m} m")
    if not 0 <= e < 1:
        raise ValueError(f"eccentricity must be in [0, 1), got {e}")
    elements = orbitalMotion.ClassicElements()
    elements.a, elements.e, elements.i = a_m, e, i_rad
    elements.Omega, elements.omega, elements.f = raan_rad, argp_rad, f_rad
    r_N, v_N = orbitalMotion.elem2rv(mu, elements)
    return np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float)
