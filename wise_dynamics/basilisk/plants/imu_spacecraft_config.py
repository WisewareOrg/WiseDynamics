"""Everything that defines one `ImuSpacecraft`."""

from dataclasses import dataclass, field

from Basilisk.utilities.macros import D2R

from wise_dynamics.basilisk.orbit_elements import OrbitElements
from wise_dynamics.basilisk.rigid_hub import RigidHub

__all__ = ["ImuSpacecraftConfig"]


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
    sigma0_BN: tuple[float, float, float] = (0.1, 0.2, -0.3)
    omega0_BN_B: tuple[float, float, float] = (1.0 * D2R, 1.0 * D2R, 1.0 * D2R)
    gyro_bias_P: tuple[float, float, float] = (1e-4, -2e-4, 5e-5)
    accel_bias_P: tuple[float, float, float] = (1e-4, -5e-5, 2e-4)
    gyro_noise_std_rad_s: float = 5e-5
    accel_noise_std_m_s2: float = 5e-5
    step_s: float = 1.0
