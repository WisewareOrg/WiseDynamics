"""A rigid spacecraft carrying one body-mounted IMU.

The vehicle is a rigid hub carrying an IMU at its centre of mass with identity mounting, constant
biases and white noise. `wise_dynamics/basilisk/README.md` tabulates its parameters with their
units and frames.
"""

from typing import Any, ClassVar

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import imuSensor
from Basilisk.simulation import spacecraft as bsk_spacecraft

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["ImuSpacecraft"]

type Vector3 = tuple[float, float, float]


class ImuSpacecraft(Vehicle):
    """A rigid spacecraft with one IMU.

    A keyword left at `None` takes the class attribute of the same name, read when the vehicle is
    built, so a consumer reads the nominal vehicle without building one and a subclass that
    overrides an attribute builds with its value. A vehicle keeps each value it is built with as
    an instance attribute of that name.

    Args:
        mass_kg: Hub mass, in kilograms.
        inertia_B_kg_m2: Hub inertia about its centre of mass in body axes, in kg*m^2, as three
            rows.
        gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P_m_s2: Constant accelerometer bias in the IMU platform frame, in m/s^2.
        gyro_noise_std_rad_s: Per-sample 1-sigma white noise on each gyro axis, in rad/s.
        accel_noise_std_m_s2: Per-sample 1-sigma white noise on each accelerometer axis, in
            m/s^2.
        noise_seed: Seed of the IMU's white-noise generator.

    Attributes:
        spacecraft: The Basilisk `Spacecraft` hub.
        imu: The Basilisk `ImuSensor`.
    """

    standard_dispersion: ClassVar[dict[str, float]] = {
        "gyro_bias_P_rad_s": 1e-4,
        "accel_bias_P_m_s2": 1e-4,
    }

    mass_kg: float = 750.0
    inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] = (
        (900.0, 0.0, 0.0),
        (0.0, 800.0, 0.0),
        (0.0, 0.0, 600.0),
    )
    gyro_bias_P_rad_s: Vector3 = (1e-4, -2e-4, 5e-5)
    accel_bias_P_m_s2: Vector3 = (1e-4, -5e-5, 2e-4)
    gyro_noise_std_rad_s: float = 5e-5
    accel_noise_std_m_s2: float = 5e-5
    noise_seed: int = 0

    def __init__(
        self,
        *,
        mass_kg: float | None = None,
        inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] | None = None,
        gyro_bias_P_rad_s: Vector3 | None = None,
        accel_bias_P_m_s2: Vector3 | None = None,
        gyro_noise_std_rad_s: float | None = None,
        accel_noise_std_m_s2: float | None = None,
        noise_seed: int | None = None,
    ) -> None:
        nominal = type(self)
        self.mass_kg = nominal.mass_kg if mass_kg is None else mass_kg
        self.inertia_B_kg_m2 = (
            nominal.inertia_B_kg_m2 if inertia_B_kg_m2 is None else inertia_B_kg_m2
        )
        self.gyro_bias_P_rad_s = (
            nominal.gyro_bias_P_rad_s if gyro_bias_P_rad_s is None else gyro_bias_P_rad_s
        )
        self.accel_bias_P_m_s2 = (
            nominal.accel_bias_P_m_s2 if accel_bias_P_m_s2 is None else accel_bias_P_m_s2
        )
        self.gyro_noise_std_rad_s = (
            nominal.gyro_noise_std_rad_s if gyro_noise_std_rad_s is None else gyro_noise_std_rad_s
        )
        self.accel_noise_std_m_s2 = (
            nominal.accel_noise_std_m_s2 if accel_noise_std_m_s2 is None else accel_noise_std_m_s2
        )
        self.noise_seed = nominal.noise_seed if noise_seed is None else noise_seed

        self.spacecraft = bsk_spacecraft.Spacecraft()
        self.spacecraft.ModelTag = "spacecraft"
        hub = self.spacecraft.hub
        hub.mHub = self.mass_kg
        hub.r_BcB_B = (0.0, 0.0, 0.0)
        hub.IHubPntBc_B = self.inertia_B_kg_m2

        self.imu = imuSensor.ImuSensor()
        self.imu.ModelTag = "imu"
        self.imu.scStateInMsg.subscribeTo(self.spacecraft.scStateOutMsg)
        self.imu.sensorPos_B = (0.0, 0.0, 0.0)
        self.imu.dcm_PB = np.eye(3)
        self.imu.senRotBias = self.gyro_bias_P_rad_s
        self.imu.senTransBias = self.accel_bias_P_m_s2
        self.imu.PMatrixGyro = self.gyro_noise_std_rad_s * np.eye(3)
        self.imu.PMatrixAccel = self.accel_noise_std_m_s2 * np.eye(3)
        self.imu.RNGSeed = int(self.noise_seed)

    def models(self) -> list[Any]:
        """The hub, then the IMU."""
        return [self.spacecraft, self.imu]

    def readers(self) -> dict[str, Any]:
        """Role `imu`: an `IMUSensorMsgReader` subscribed to the IMU's output."""
        imu_reader = messaging.IMUSensorMsgReader()
        imu_reader.subscribeTo(self.imu.sensorOutMsg)
        return {"imu": imu_reader}

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """The hub's `SCStatesMsg` attitude, rate, position and velocity under `truth/`, and the
        IMU's `IMUSensorMsg` gyro and accelerometer outputs under `imu/`.

        `truth/r_N` and `truth/v_N` are the body-frame origin's state in the inertial frame N. They
        equal the centre of mass's state, which a scenario places, because the hub's centre of
        mass sits at the body-frame origin.
        """
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
