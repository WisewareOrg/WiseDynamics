"""The IMU spacecraft plant: its Basilisk hub, gravity and IMU."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import imuSensor
from Basilisk.simulation import spacecraft as bsk_spacecraft
from Basilisk.utilities import simHelpers, simIncludeGravBody
from Basilisk.utilities.macros import D2R

from wise_dynamics.basilisk.plant import Plant
from wise_dynamics.basilisk.plants.imu_spacecraft.config import ImuSpacecraftConfig
from wise_dynamics.basilisk.recorded_field import RecordedField
from wise_dynamics.basilisk.swig import Message, MessageReader
from wise_dynamics.basilisk.vectors import column
from wise_dynamics.dispersion import Dispersion

__all__ = ["ImuSpacecraft"]


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
        hub_object.hub.r_BcB_B = column((0.0, 0.0, 0.0))
        hub_object.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
            [element for row in config.hub.inertia_B_kg_m2 for element in row]
        )
        hub_object.hub.sigma_BNInit = column(config.sigma0_BN)
        hub_object.hub.omega_BN_BInit = column(config.omega0_BN_B)
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
        imu.sensorPos_B = column((0.0, 0.0, 0.0))
        imu.dcm_PB = np.eye(3).tolist()
        imu.senRotBias = column(config.gyro_bias_P)
        imu.senTransBias = column(config.accel_bias_P)
        imu.PMatrixGyro = (config.gyro_noise_std_rad_s * np.eye(3)).tolist()
        imu.PMatrixAccel = (config.accel_noise_std_m_s2 * np.eye(3)).tolist()
        imu.RNGSeed = int(noise_seed)
        self.add_model(imu)
        return imu
