"""A rigid spacecraft in low Earth orbit carrying one body-mounted IMU.

The vehicle is a rigid hub under point-mass Earth gravity, started on a classical-element orbit
with an initial attitude and body rate, and carries an IMU at its centre of mass with identity
mounting, constant biases and white noise. `wise_dynamics/basilisk/README.md` tabulates its
parameters with their units and frames.
"""

from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import imuSensor
from Basilisk.simulation import spacecraft as bsk_spacecraft
from Basilisk.utilities import orbitalMotion, simHelpers, simIncludeGravBody
from Basilisk.utilities.macros import D2R

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["ImuSpacecraft"]

type Vector3 = tuple[float, float, float]


class ImuSpacecraft(Vehicle):
    """A tumbling rigid spacecraft in low Earth orbit with one IMU.

    Each keyword defaults to the class attribute of the same name, the nominal vehicle, so a
    consumer reads a default without building a vehicle; a vehicle keeps each keyword it is built
    with as an instance attribute of that name.

    Args:
        semi_major_axis_m: Semi-major axis of the initial orbit, in metres.
        eccentricity: Eccentricity of the initial orbit, `0 <= e < 1`.
        inclination_rad: Inclination, in radians.
        raan_rad: Right ascension of the ascending node, in radians.
        arg_periapsis_rad: Argument of periapsis, in radians.
        true_anomaly_rad: True anomaly at t = 0, in radians.
        mass_kg: Hub mass, in kilograms.
        inertia_B_kg_m2: Hub inertia about its centre of mass in body axes, in kg*m^2, as three
            rows.
        sigma0_BN: Initial attitude, the MRP of the body frame relative to inertial.
        omega0_BN_B_rad_s: Initial body rate relative to inertial, in body axes, in rad/s.
        gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P_m_s2: Constant accelerometer bias in the IMU platform frame, in m/s^2.
        gyro_noise_std_rad_s: Per-sample 1-sigma white noise on each gyro axis, in rad/s.
        accel_noise_std_m_s2: Per-sample 1-sigma white noise on each accelerometer axis, in
            m/s^2.
        step_s: The dynamics and sensor step, in seconds.
        noise_seed: Seed of the IMU's white-noise generator.

    Attributes:
        spacecraft: The Basilisk `Spacecraft` hub.
        imu: The Basilisk `ImuSensor`.

    Raises:
        ValueError: `semi_major_axis_m` or `step_s` is not positive, or `eccentricity` is outside
            [0, 1).
    """

    mu: ClassVar[float] = 3.986004360e14
    """Earth's gravitational parameter in Basilisk's point-mass gravity model, in m^3/s^2."""

    standard_dispersion: ClassVar[MappingProxyType[str, float]] = MappingProxyType(
        {
            "sigma0_BN": 0.01,
            "omega0_BN_B_rad_s": 0.2 * D2R,
            "gyro_bias_P_rad_s": 1e-4,
            "accel_bias_P_m_s2": 1e-4,
        }
    )

    semi_major_axis_m: float = 6871.0e3
    eccentricity: float = 0.0001
    inclination_rad: float = 45.0 * D2R
    raan_rad: float = 48.2 * D2R
    arg_periapsis_rad: float = 347.8 * D2R
    true_anomaly_rad: float = 85.3 * D2R
    mass_kg: float = 750.0
    inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] = (
        (900.0, 0.0, 0.0),
        (0.0, 800.0, 0.0),
        (0.0, 0.0, 600.0),
    )
    sigma0_BN: Vector3 = (0.1, 0.2, -0.3)
    omega0_BN_B_rad_s: Vector3 = (1.0 * D2R, 1.0 * D2R, 1.0 * D2R)
    gyro_bias_P_rad_s: Vector3 = (1e-4, -2e-4, 5e-5)
    accel_bias_P_m_s2: Vector3 = (1e-4, -5e-5, 2e-4)
    gyro_noise_std_rad_s: float = 5e-5
    accel_noise_std_m_s2: float = 5e-5
    step_s: float = 1.0
    noise_seed: int = 0

    def __init__(
        self,
        *,
        semi_major_axis_m: float = semi_major_axis_m,
        eccentricity: float = eccentricity,
        inclination_rad: float = inclination_rad,
        raan_rad: float = raan_rad,
        arg_periapsis_rad: float = arg_periapsis_rad,
        true_anomaly_rad: float = true_anomaly_rad,
        mass_kg: float = mass_kg,
        inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] = inertia_B_kg_m2,
        sigma0_BN: Vector3 = sigma0_BN,
        omega0_BN_B_rad_s: Vector3 = omega0_BN_B_rad_s,
        gyro_bias_P_rad_s: Vector3 = gyro_bias_P_rad_s,
        accel_bias_P_m_s2: Vector3 = accel_bias_P_m_s2,
        gyro_noise_std_rad_s: float = gyro_noise_std_rad_s,
        accel_noise_std_m_s2: float = accel_noise_std_m_s2,
        step_s: float = step_s,
        noise_seed: int = noise_seed,
    ) -> None:
        self.semi_major_axis_m = semi_major_axis_m
        self.eccentricity = eccentricity
        self.inclination_rad = inclination_rad
        self.raan_rad = raan_rad
        self.arg_periapsis_rad = arg_periapsis_rad
        self.true_anomaly_rad = true_anomaly_rad
        self.mass_kg = mass_kg
        self.inertia_B_kg_m2 = inertia_B_kg_m2
        self.sigma0_BN = sigma0_BN
        self.omega0_BN_B_rad_s = omega0_BN_B_rad_s
        self.gyro_bias_P_rad_s = gyro_bias_P_rad_s
        self.accel_bias_P_m_s2 = accel_bias_P_m_s2
        self.gyro_noise_std_rad_s = gyro_noise_std_rad_s
        self.accel_noise_std_m_s2 = accel_noise_std_m_s2
        self.noise_seed = noise_seed
        r_N, v_N = _orbit_to_inertial(
            self.mu,
            self.semi_major_axis_m,
            self.eccentricity,
            self.inclination_rad,
            self.raan_rad,
            self.arg_periapsis_rad,
            self.true_anomaly_rad,
        )
        super().__init__(step_s=step_s)

        self.spacecraft = bsk_spacecraft.Spacecraft()
        self.spacecraft.ModelTag = "spacecraft"
        hub = self.spacecraft.hub
        hub.mHub = self.mass_kg
        hub.r_BcB_B = _column((0.0, 0.0, 0.0))
        hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
            [element for row in self.inertia_B_kg_m2 for element in row]
        )
        hub.sigma_BNInit = _column(self.sigma0_BN)
        hub.omega_BN_BInit = _column(self.omega0_BN_B_rad_s)
        self.add_model(self.spacecraft)

        self._gravity = simIncludeGravBody.gravBodyFactory()
        earth = self._gravity.createEarth()
        earth.isCentralBody = True
        self._gravity.addBodiesTo(self.spacecraft)
        hub.r_CN_NInit = r_N.tolist()
        hub.v_CN_NInit = v_N.tolist()

        self.imu = imuSensor.ImuSensor()
        self.imu.ModelTag = "imu"
        self.imu.scStateInMsg.subscribeTo(self.spacecraft.scStateOutMsg)
        self.imu.sensorPos_B = _column((0.0, 0.0, 0.0))
        self.imu.dcm_PB = np.eye(3).tolist()
        self.imu.senRotBias = _column(self.gyro_bias_P_rad_s)
        self.imu.senTransBias = _column(self.accel_bias_P_m_s2)
        self.imu.PMatrixGyro = (self.gyro_noise_std_rad_s * np.eye(3)).tolist()
        self.imu.PMatrixAccel = (self.accel_noise_std_m_s2 * np.eye(3)).tolist()
        self.imu.RNGSeed = int(self.noise_seed)
        self.add_model(self.imu)

    def _readers(self) -> dict[str, Any]:
        """Role `imu`: an `IMUSensorMsgReader` subscribed to the IMU's output."""
        imu_reader = messaging.IMUSensorMsgReader()
        imu_reader.subscribeTo(self.imu.sensorOutMsg)
        return {"imu": imu_reader}

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """The hub's `SCStatesMsg` attitude, rate, position and velocity under `truth/`, and the
        IMU's `IMUSensorMsg` gyro and accelerometer outputs under `imu/`."""
        truth = self.spacecraft.scStateOutMsg
        imu = self.imu.sensorOutMsg
        return {
            "truth/sigma_BN": (truth, "sigma_BN", "MRP"),
            "truth/omega_BN_B": (truth, "omega_BN_B", "rad/s"),
            "truth/r_N": (truth, "r_BN_N", "m"),
            "truth/v_N": (truth, "v_BN_N", "m/s"),
            "imu/gyro_P": (imu, "AngVelPlatform", "rad/s"),
            "imu/accel_P": (imu, "AccelPlatform", "m/s^2"),
        }

    def initial_state(self) -> dict[str, np.ndarray]:
        """The hub's initial condition, each a float array of shape `(3,)`.

        Returns:
            `sigma0_BN` (MRP, body relative to inertial), `r0_N` (inertial position of the
            centre of mass, m), `v0_N` (its inertial velocity, m/s) and `omega0_BN_B` (body
            rate relative to inertial in body axes, rad/s).
        """
        hub = self.spacecraft.hub
        return {
            "sigma0_BN": np.array(hub.sigma_BNInit, dtype=float).flatten(),
            "r0_N": np.array(hub.r_CN_NInit, dtype=float).flatten(),
            "v0_N": np.array(hub.v_CN_NInit, dtype=float).flatten(),
            "omega0_BN_B": np.array(hub.omega_BN_BInit, dtype=float).flatten(),
        }


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


def _column(vector: Vector3) -> list[list[float]]:
    """A 3-vector in Basilisk's column form, `[[x], [y], [z]]`."""
    return [[float(component)] for component in vector]
