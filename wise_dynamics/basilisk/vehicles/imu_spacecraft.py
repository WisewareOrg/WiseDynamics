"""A rigid spacecraft carrying one body-mounted IMU.

The vehicle is a rigid hub carrying an IMU at its centre of mass with identity mounting, constant
biases and white noise specified as a density.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import imuSensor
from Basilisk.simulation import spacecraft as bsk_spacecraft
from scipy.spatial.transform import Rotation

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["ImuSpacecraft"]

type Vector3 = tuple[float, float, float]


class ImuSpacecraft(Vehicle):
    """A rigid hub with one IMU at its centre of mass and identity mounting.

    The IMU platform frame P is the body frame B. A keyword left at `None` takes the class
    attribute of the same name, read when the vehicle is built, so a consumer reads the nominal
    vehicle without building one and a subclass that overrides an attribute builds with its value.
    A vehicle keeps each value it is built with as an instance attribute of that name, and holds
    `spacecraft`, the Basilisk `Spacecraft` hub, and `imu`, the Basilisk `ImuSensor`.

    Attributes:
        mass_kg: Nominal hub mass, in kilograms.
        inertia_B_kg_m2: Nominal hub inertia about its centre of mass in body axes, in kg*m^2, as
            three rows.
        gyro_bias_P_rad_s: Nominal constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P_m_s2: Nominal constant accelerometer bias in the IMU platform frame, in
            m/s^2.
        gyro_noise_density_rad_s_rthz: Nominal white-noise density on each gyro axis, in
            rad/s/sqrt(Hz).
        accel_noise_density_m_s2_rthz: Nominal white-noise density on each accelerometer axis, in
            m/s^2/sqrt(Hz).
        noise_seed: Nominal seed of the IMU's white-noise generator.
        gyro_bias_std_P_rad_s: Per-axis 1-sigma of the gyro bias `dispersed` draws, in rad/s.
        accel_bias_std_P_m_s2: Per-axis 1-sigma of the accelerometer bias `dispersed` draws, in
            m/s^2.
    """

    mass_kg: float = 750.0
    inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] = (
        (900.0, 0.0, 0.0),
        (0.0, 800.0, 0.0),
        (0.0, 0.0, 600.0),
    )
    gyro_bias_P_rad_s: Vector3 = (1e-4, -2e-4, 5e-5)
    accel_bias_P_m_s2: Vector3 = (1e-4, -5e-5, 2e-4)
    gyro_noise_density_rad_s_rthz: float = 5e-5
    accel_noise_density_m_s2_rthz: float = 5e-5
    noise_seed: int = 0
    gyro_bias_std_P_rad_s: float = 1e-4
    accel_bias_std_P_m_s2: float = 1e-4

    @classmethod
    def dispersed(
        cls, rng: np.random.Generator, *, noise_seed: int
    ) -> tuple["ImuSpacecraft", dict[str, np.ndarray]]:
        """A vehicle built with biases drawn from `rng`, and what was drawn.

        Each bias is its nominal plus independent normal samples per axis of its 1-sigma, the gyro
        bias first, then the accelerometer bias.

        Args:
            rng: The generator the biases are drawn from.
            noise_seed: The vehicle's sensor-noise seed.

        Returns:
            The vehicle, and `gyro_bias_P_rad_s` and `accel_bias_P_m_s2` as drawn.
        """
        draws = {
            "gyro_bias_P_rad_s": np.asarray(cls.gyro_bias_P_rad_s)
            + rng.normal(0.0, cls.gyro_bias_std_P_rad_s, size=3),
            "accel_bias_P_m_s2": np.asarray(cls.accel_bias_P_m_s2)
            + rng.normal(0.0, cls.accel_bias_std_P_m_s2, size=3),
        }
        return cls(**draws, noise_seed=noise_seed), draws

    def __init__(
        self,
        *,
        mass_kg: float | None = None,
        inertia_B_kg_m2: tuple[Vector3, Vector3, Vector3] | None = None,
        gyro_bias_P_rad_s: Vector3 | None = None,
        accel_bias_P_m_s2: Vector3 | None = None,
        gyro_noise_density_rad_s_rthz: float | None = None,
        accel_noise_density_m_s2_rthz: float | None = None,
        noise_seed: int | None = None,
    ) -> None:
        """Build the hub and the IMU.

        Args:
            mass_kg: Hub mass, in kilograms.
            inertia_B_kg_m2: Hub inertia about its centre of mass in body axes, in kg*m^2, as
                three rows.
            gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
            accel_bias_P_m_s2: Constant accelerometer bias in the IMU platform frame, in m/s^2.
            gyro_noise_density_rad_s_rthz: White-noise density on each gyro axis, in
                rad/s/sqrt(Hz).
            accel_noise_density_m_s2_rthz: White-noise density on each accelerometer axis, in
                m/s^2/sqrt(Hz).
            noise_seed: Seed of the IMU's white-noise generator.
        """
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
        self.gyro_noise_density_rad_s_rthz = (
            nominal.gyro_noise_density_rad_s_rthz
            if gyro_noise_density_rad_s_rthz is None
            else gyro_noise_density_rad_s_rthz
        )
        self.accel_noise_density_m_s2_rthz = (
            nominal.accel_noise_density_m_s2_rthz
            if accel_noise_density_m_s2_rthz is None
            else accel_noise_density_m_s2_rthz
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
        self.imu.RNGSeed = int(self.noise_seed)

    def models(self) -> list[Any]:
        """List the vehicle's propagated Basilisk modules in task order.

        Returns:
            The hub.
        """
        return [self.spacecraft]

    def sensor_models(self, sample_period_s: float) -> list[Any]:
        """Set each sensor's per-sample noise for its sample period, and list the sensors.

        A sample carries white noise of 1-sigma density / sqrt(`sample_period_s`) on each axis.

        Args:
            sample_period_s: The period the sensors sample at, in seconds.

        Returns:
            The IMU.
        """
        root_period = np.sqrt(sample_period_s)
        self.imu.PMatrixGyro = self.gyro_noise_density_rad_s_rthz / root_period * np.eye(3)
        self.imu.PMatrixAccel = self.accel_noise_density_m_s2_rthz / root_period * np.eye(3)
        return [self.imu]

    def readers(self) -> dict[str, Any]:
        """Build the vehicle's sensor readers.

        Returns:
            Role `imu`: an `IMUSensorMsgReader` subscribed to the IMU's output.
        """
        imu_reader = messaging.IMUSensorMsgReader()
        imu_reader.subscribeTo(self.imu.sensorOutMsg)
        return {"imu": imu_reader}

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Declare what the vehicle records.

        `truth/r_N` and `truth/v_N` are the body-frame origin's state in the inertial frame N. They
        equal the centre of mass's state, which a scenario places, because the hub's centre of
        mass sits at the body-frame origin.

        Returns:
            Run-file dataset path to `(message, field, units)`: the hub's `SCStatesMsg` attitude
            (Basilisk's MRP, which `finish` turns into `truth/q_BN`), rate (rad/s), position (m)
            and velocity (m/s) under `truth/`, and the IMU's `IMUSensorMsg` gyro (rad/s) and
            accelerometer (m/s^2) outputs under `imu/`.
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

    def finish(self, datasets: Mapping[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
        """Replace the recorded attitude MRP with the quaternion, converted in one batch.

        Args:
            datasets: Dataset name to `(array, units)`, as the recorders captured them.

        Returns:
            The datasets with `truth/sigma_BN` replaced, in its place, by `truth/q_BN`: (N, 4)
            quaternions W X Y Z of the body frame relative to N (Hamilton, body to N), W >= 0.
        """
        finished = {}
        for name, (array, units) in datasets.items():
            if name == "truth/sigma_BN":
                quaternion = Rotation.from_mrp(array).as_quat(canonical=True, scalar_first=True)
                finished["truth/q_BN"] = (quaternion, "")
            else:
                finished[name] = (array, units)
        return finished

    def tracking(self) -> Callable[[], dict[str, np.ndarray]]:
        """A stand-in for tracking: the vehicle's true state, as a flight software never sees it.

        Returns:
            A function giving the state as of the last time the hub's state was written: `q_BN`
            (the quaternion W X Y Z, W >= 0), `r_N` (m) and `v_N` (m/s), each from the hub's
            `SCStatesMsg`.
        """
        reader = messaging.SCStatesMsgReader()
        reader.subscribeTo(self.spacecraft.scStateOutMsg)

        def read() -> dict[str, np.ndarray]:
            """The tracked state now.

            Returns:
                `q_BN`, `r_N` and `v_N` from the hub's last `SCStatesMsg`.
            """
            state = reader()
            q = Rotation.from_mrp(state.sigma_BN).as_quat(canonical=True, scalar_first=True)
            return {"q_BN": q, "r_N": np.array(state.r_BN_N), "v_N": np.array(state.v_BN_N)}

        return read
