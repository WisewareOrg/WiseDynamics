"""A rigid spacecraft in low Earth orbit carrying one body-mounted IMU.

The vehicle is a rigid hub under point-mass Earth gravity, started on a classical-element orbit
with an initial attitude and body rate, and carries an IMU at its centre of mass with identity
mounting, constant biases and white noise. Units and frames are in each field's name or
docstring; `wise_dynamics/basilisk/README.md` tabulates them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import imuSensor
from Basilisk.simulation import spacecraft as bsk_spacecraft
from Basilisk.utilities import orbitalMotion, simHelpers, simIncludeGravBody
from Basilisk.utilities.macros import D2R

from wise_dynamics.basilisk.plant import Message, MessageReader, Plant, RecordedField
from wise_dynamics.dispersion import Dispersion

__all__ = ["ImuSpacecraft", "ImuSpacecraftConfig", "OrbitElements", "RigidHub"]

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]


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


@dataclass(frozen=True, kw_only=True)
class RigidHub:
    """The spacecraft's rigid hub.

    Attributes:
        mass_kg: Mass, in kilograms.
        inertia_B_kg_m2: Inertia tensor about the centre of mass in body axes, in kg*m^2, as
            three rows.
    """

    mass_kg: float = 750.0
    inertia_B_kg_m2: Matrix3 = ((900.0, 0.0, 0.0), (0.0, 800.0, 0.0), (0.0, 0.0, 600.0))


@dataclass(frozen=True, kw_only=True)
class ImuSpacecraftConfig:
    """Everything that defines one `ImuSpacecraft`. The defaults are the nominal vehicle.

    Attributes:
        orbit: The initial orbit.
        hub: The rigid hub's mass properties.
        sigma0_BN: Initial attitude, the MRP of the body frame relative to inertial.
        omega0_BN_B: Initial body rate relative to inertial, in body axes, in rad/s.
        gyro_bias_P: Constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P: Constant accelerometer bias in the IMU platform frame, in m/s^2.
        gyro_noise_std_rad_s: Per-sample 1-sigma white noise on each gyro axis, in rad/s.
        accel_noise_std_m_s2: Per-sample 1-sigma white noise on each accelerometer axis, in
            m/s^2.
        step_s: The dynamics and sensor step, in seconds.
    """

    orbit: OrbitElements = field(default_factory=OrbitElements)
    hub: RigidHub = field(default_factory=RigidHub)
    sigma0_BN: Vector3 = (0.1, 0.2, -0.3)
    omega0_BN_B: Vector3 = (1.0 * D2R, 1.0 * D2R, 1.0 * D2R)
    gyro_bias_P: Vector3 = (1e-4, -2e-4, 5e-5)
    accel_bias_P: Vector3 = (1e-4, -5e-5, 2e-4)
    gyro_noise_std_rad_s: float = 5e-5
    accel_noise_std_m_s2: float = 5e-5
    step_s: float = 1.0


class ImuSpacecraft(Plant[ImuSpacecraftConfig]):
    """A tumbling rigid spacecraft in low Earth orbit with one IMU.

    Args:
        config: The vehicle; `None` selects `nominal()`.
        noise_seed: Seed of the IMU's white-noise generator.

    Attributes:
        spacecraft: The Basilisk `Spacecraft` hub.
        imu: The Basilisk `ImuSensor`.
    """

    mu: ClassVar[float] = 3.986004360e14
    """Earth's gravitational parameter, in m^3/s^2: the point-mass gravity model's and the orbit
    conversion's."""

    standard_dispersion: ClassVar[Dispersion] = Dispersion(
        {
            "sigma0_BN": 0.01,
            "omega0_BN_B": 0.2 * D2R,
            "gyro_bias_P": 1e-4,
            "accel_bias_P": 1e-4,
        }
    )
    recorded_fields: ClassVar[Mapping[str, RecordedField]] = MappingProxyType(
        {
            "truth/sigma_BN": RecordedField("truth", "sigma_BN", "MRP"),
            "truth/omega_BN_B": RecordedField("truth", "omega_BN_B", "rad/s"),
            "truth/r_N": RecordedField("truth", "r_BN_N", "m"),
            "truth/v_N": RecordedField("truth", "v_BN_N", "m/s"),
            "imu/gyro_P": RecordedField("imu", "AngVelPlatform", "rad/s"),
            "imu/accel_P": RecordedField("imu", "AccelPlatform", "m/s^2"),
        }
    )

    spacecraft: bsk_spacecraft.Spacecraft
    imu: imuSensor.ImuSensor

    def _build(self, config: ImuSpacecraftConfig, noise_seed: int) -> None:
        self.spacecraft = self._build_hub(config)
        self._build_gravity()
        r_N, v_N = config.orbit.to_inertial(self.mu)
        self.spacecraft.hub.r_CN_NInit = r_N.tolist()
        self.spacecraft.hub.v_CN_NInit = v_N.tolist()
        self.imu = self._build_imu(config, noise_seed)

    @classmethod
    def nominal(cls) -> ImuSpacecraftConfig:
        """The nominal vehicle, `ImuSpacecraftConfig()`."""
        return ImuSpacecraftConfig()

    def _readers(self) -> dict[str, MessageReader]:
        """Role `imu`: an `IMUSensorMsgReader` subscribed to the IMU's output."""
        imu_reader = messaging.IMUSensorMsgReader()
        imu_reader.subscribeTo(self.imu.sensorOutMsg)
        return {"imu": imu_reader}

    def truth_messages(self) -> dict[str, Message]:
        """Role `truth`, the hub's `SCStatesMsg`, and role `imu`, the IMU's `IMUSensorMsg`."""
        return {"truth": self.spacecraft.scStateOutMsg, "imu": self.imu.sensorOutMsg}

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

    def _build_hub(self, config: ImuSpacecraftConfig) -> bsk_spacecraft.Spacecraft:
        hub_object = bsk_spacecraft.Spacecraft()
        hub_object.ModelTag = "spacecraft"
        hub_object.hub.mHub = config.hub.mass_kg
        hub_object.hub.r_BcB_B = _column((0.0, 0.0, 0.0))
        hub_object.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
            [element for row in config.hub.inertia_B_kg_m2 for element in row]
        )
        hub_object.hub.sigma_BNInit = _column(config.sigma0_BN)
        hub_object.hub.omega_BN_BInit = _column(config.omega0_BN_B)
        self.add_model(hub_object)
        return hub_object

    def _build_gravity(self) -> None:
        self._gravity = simIncludeGravBody.gravBodyFactory()
        earth = self._gravity.createEarth()
        earth.isCentralBody = True
        earth.mu = self.mu
        self._gravity.addBodiesTo(self.spacecraft)

    def _build_imu(self, config: ImuSpacecraftConfig, noise_seed: int) -> imuSensor.ImuSensor:
        imu = imuSensor.ImuSensor()
        imu.ModelTag = "imu"
        imu.scStateInMsg.subscribeTo(self.spacecraft.scStateOutMsg)
        imu.sensorPos_B = _column((0.0, 0.0, 0.0))
        imu.dcm_PB = np.eye(3).tolist()
        imu.senRotBias = _column(config.gyro_bias_P)
        imu.senTransBias = _column(config.accel_bias_P)
        imu.PMatrixGyro = (config.gyro_noise_std_rad_s * np.eye(3)).tolist()
        imu.PMatrixAccel = (config.accel_noise_std_m_s2 * np.eye(3)).tolist()
        imu.RNGSeed = int(noise_seed)
        self.add_model(imu)
        return imu


def _column(vector: Vector3) -> list[list[float]]:
    """A 3-vector in Basilisk's column form, `[[x], [y], [z]]`."""
    return [[float(component)] for component in vector]
