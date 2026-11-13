"""An IMU spacecraft in a 500 km, 45 degree low Earth orbit under point-mass Earth gravity."""

import numpy as np
from Basilisk.utilities import orbitalMotion, simIncludeGravBody
from Basilisk.utilities.macros import D2R
from scipy.spatial.transform import Rotation

from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft

__all__ = ["Leo500"]

_ATTITUDE_ROTVEC_PER_DRAW = 4.0 / 1.14
"""Rotation vector, in rad, per unit of the attitude draw, so a draw of 1-sigma 0.01 turns the body
by 0.0351 rad per axis: the spread of an additive MRP step of 0.01 per axis about the nominal MRP
(0.1, 0.2, -0.3). Such a step is the body rotation `4 B(sigma)^-1 d_sigma`, and
`B^T B = (1 + |sigma|^2)^2 I` with `|sigma|^2 = 0.14` makes it isotropic at 4 / 1.14 times the
step."""


class Leo500(Scenario):
    """A vehicle tumbling on a classical-element orbit about a point-mass Earth.

    The inertial frame N is Earth-centred, with no rotation and no epoch: point-mass gravity has
    nothing that depends on time. The class attributes define the scenario, and the constructor
    builds the vehicle and places it there.

    Attributes:
        mu_m3_s2: Earth's gravitational parameter, in m^3/s^2, keyed `"earth"`.
        semi_major_axis_m: Semi-major axis of the initial orbit, in metres.
        eccentricity: Eccentricity of the initial orbit.
        inclination_rad: Inclination, in radians.
        raan_rad: Right ascension of the ascending node, in radians.
        arg_periapsis_rad: Argument of periapsis, in radians.
        true_anomaly_rad: True anomaly at t = 0, in radians.
        q0_BN: Initial attitude, the unit quaternion W X Y Z of the body frame relative to N
            (Hamilton, body to N): the one whose MRP is (0.1, 0.2, -0.3).
        omega0_BN_B_rad_s: Initial body rate relative to N, in body axes, in rad/s.
    """

    mu_m3_s2: dict[str, float] = {"earth": 3.986004360e14}
    semi_major_axis_m: float = 6871.0e3
    eccentricity: float = 0.0001
    inclination_rad: float = 45.0 * D2R
    raan_rad: float = 48.2 * D2R
    arg_periapsis_rad: float = 347.8 * D2R
    true_anomaly_rad: float = 85.3 * D2R
    q0_BN: tuple[float, float, float, float] = (
        0.7543859649122805,
        0.17543859649122806,
        0.3508771929824561,
        -0.5263157894736842,
    )
    omega0_BN_B_rad_s: tuple[float, float, float] = (1.0 * D2R, 1.0 * D2R, 1.0 * D2R)

    def __init__(self, vehicle_type: type[ImuSpacecraft] = ImuSpacecraft) -> None:
        """Build a vehicle of `vehicle_type` and place it at the class attributes' placement.

        The scenario holds it as `vehicle`, which `disperse` replaces, and in `vehicles`, as
        `[vehicle]`. `draws`, what `disperse` drew and applied, name to value, starts empty.

        Args:
            vehicle_type: The vehicle to fly, an `ImuSpacecraft` or a subclass of it.
        """
        super().__init__()
        self.draws: dict[str, np.ndarray] = {}
        self.vehicle = self._placed(vehicle_type())
        self.vehicles = [self.vehicle]

    def disperse(self, rng: np.random.Generator, *, noise_seed: int) -> None:
        """Disperse this case: the placement first, then the vehicle's own dispersion.

        The vehicle class draws its own dispersion from the same `rng` and builds the vehicle,
        which is then placed. The placement is `q0_BN` turned about the body axes by a rotation
        vector, one `rng.normal(0.0, 0.01, size=3)` draw scaled by `_ATTITUDE_ROTVEC_PER_DRAW`,
        then `omega0_BN_B_rad_s` with 0.2 deg/s, its nominal plus independent normal samples. The
        scale is fixed, so a subclass that overrides `q0_BN` keeps the same spread. The new vehicle
        replaces `vehicle`, so readers and other handles on it are taken after dispersing. `draws`
        records the placement's draws and the vehicle's.

        Args:
            rng: The generator the dispersions are drawn from.
            noise_seed: The new vehicle's sensor-noise seed.
        """
        turn = Rotation.from_rotvec(rng.normal(0.0, 0.01, size=3) * _ATTITUDE_ROTVEC_PER_DRAW)
        nominal = Rotation.from_quat(type(self).q0_BN, scalar_first=True)
        self.q0_BN = (nominal * turn).as_quat(canonical=True, scalar_first=True)
        self.omega0_BN_B_rad_s = np.asarray(type(self).omega0_BN_B_rad_s) + rng.normal(
            0.0, 0.2 * D2R, size=3
        )
        vehicle, vehicle_draws = type(self.vehicle).dispersed(rng, noise_seed=noise_seed)
        self.vehicle = self._placed(vehicle)
        self.vehicles = [self.vehicle]
        self.draws = {
            "q0_BN": self.q0_BN,
            "omega0_BN_B_rad_s": self.omega0_BN_B_rad_s,
            **vehicle_draws,
        }

    def initial_state(self) -> dict[str, np.ndarray]:
        """Compute the initial condition the placement gives the vehicle.

        Returns:
            `q0_BN` (unit quaternion W X Y Z, body relative to N, shape `(4,)`), `r0_N` (position
            of the centre of mass in N, m), `v0_N` (its velocity in N, m/s) and `omega0_BN_B` (body
            rate relative to N in body axes, rad/s), each a float array, `(3,)` unless noted.
        """
        elements = orbitalMotion.ClassicElements()
        elements.a, elements.e = self.semi_major_axis_m, self.eccentricity
        elements.i, elements.Omega = self.inclination_rad, self.raan_rad
        elements.omega, elements.f = self.arg_periapsis_rad, self.true_anomaly_rad
        r_N, v_N = orbitalMotion.elem2rv(self.mu_m3_s2["earth"], elements)
        return {
            "q0_BN": np.asarray(self.q0_BN, dtype=float),
            "r0_N": np.asarray(r_N, dtype=float),
            "v0_N": np.asarray(v_N, dtype=float),
            "omega0_BN_B": np.asarray(self.omega0_BN_B_rad_s, dtype=float),
        }

    def _placed(self, vehicle: ImuSpacecraft) -> ImuSpacecraft:
        """Place a vehicle under point-mass Earth gravity, at the placement's initial state.

        Args:
            vehicle: The vehicle to place.

        Returns:
            `vehicle`, with Earth attached as its central body and the initial state on its hub.
        """
        state = self.initial_state()
        self._gravity = simIncludeGravBody.gravBodyFactory()
        earth = self._gravity.createEarth()
        earth.mu = self.mu_m3_s2["earth"]
        earth.isCentralBody = True
        self._gravity.addBodiesTo(vehicle.spacecraft)

        hub = vehicle.spacecraft.hub
        hub.r_CN_NInit = state["r0_N"]
        hub.v_CN_NInit = state["v0_N"]
        hub.sigma_BNInit = Rotation.from_quat(state["q0_BN"], scalar_first=True).as_mrp()
        hub.omega_BN_BInit = state["omega0_BN_B"]
        return vehicle
