"""A reusable Basilisk dynamics stack, bskSim-style: a rigid spacecraft hub under point-mass
gravity on a classical-element orbit, with an IMU sensor attached."""

import numpy as np
from Basilisk.simulation import imuSensor, spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simHelpers,
    simIncludeGravBody,
)

# Nominal demo configuration: 500 km circular LEO at 45 deg inclination, Earth point-mass
# gravity, a tumbling rigid hub.
ORBIT_A = 6871.0e3  # m
ORBIT_E = 0.0001  # -
ORBIT_I = 45.0 * macros.D2R  # rad
ORBIT_OMEGA_CAP = 48.2 * macros.D2R  # rad, right ascension of ascending node
ORBIT_OMEGA_LOW = 347.8 * macros.D2R  # rad, argument of periapsis
ORBIT_F = 85.3 * macros.D2R  # rad, true anomaly

HUB_MASS = 750.0  # kg
HUB_INERTIA = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]  # kg*m^2, diagonal

SIGMA0_BN_DEFAULT = np.array([0.1, 0.2, -0.3])  # -, MRP
OMEGA0_BN_B_DEFAULT = np.array([1.0, 1.0, 1.0]) * macros.D2R  # rad/s per axis

# IMU error model defaults (applied to the identity-mounted sensor; modest white noise on top of
# the constant bias, zero correlation across axes).
GYRO_BIAS_DEFAULT = np.array([1e-4, -2e-4, 5e-5])  # rad/s
ACCEL_BIAS_DEFAULT = np.array([1e-4, -5e-5, 2e-4])  # m/s^2
GYRO_NOISE_PSD_DEFAULT = 1e-12  # (rad/s)^2 * s, diagonal PMatrixGyro
ACCEL_NOISE_PSD_DEFAULT = 1e-10  # (m/s^2)^2 * s, diagonal PMatrixAccel

DYNAMICS_STEP_S = 1.0  # s, dynamics/sensor/algorithm step


def _col(v: np.ndarray) -> list[list[float]]:
    """Converts a 3-vector to Basilisk's column-vector list form `[[x], [y], [z]]`."""
    return [[float(x)] for x in v]


class ImuSpacecraftModel:
    """Builds the process, task, hub, point-mass Earth, orbit, and IMU on `sim` (a
    `SimBaseClass`), exposing `scObject`, `imuObject`, `mu`, `task_name`, and `sim` as
    attributes a caller addresses directly, by dotted path or otherwise."""

    def __init__(
        self,
        sim: SimulationBaseClass.SimBaseClass,
        task_name: str = "simTask",
        process_name: str = "simProcess",
        step_s: float = DYNAMICS_STEP_S,
        orbit_a: float = ORBIT_A,
        orbit_e: float = ORBIT_E,
        orbit_i: float = ORBIT_I,
        orbit_Omega: float = ORBIT_OMEGA_CAP,
        orbit_omega: float = ORBIT_OMEGA_LOW,
        orbit_f: float = ORBIT_F,
        hub_mass: float = HUB_MASS,
        hub_inertia: list[float] = HUB_INERTIA,
        sigma0_BN: np.ndarray = SIGMA0_BN_DEFAULT,
        omega0_BN_B: np.ndarray = OMEGA0_BN_B_DEFAULT,
        gyro_bias: np.ndarray = GYRO_BIAS_DEFAULT,
        accel_bias: np.ndarray = ACCEL_BIAS_DEFAULT,
        gyro_noise_psd: float = GYRO_NOISE_PSD_DEFAULT,
        accel_noise_psd: float = ACCEL_NOISE_PSD_DEFAULT,
    ) -> None:
        self.sim = sim
        self.task_name = task_name

        self.dynProcess = sim.CreateNewProcess(process_name)
        self.dynProcess.addTask(sim.CreateNewTask(task_name, macros.sec2nano(step_s)))

        # Rigid hub under point-mass gravity alone.
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        self.scObject.hub.mHub = hub_mass
        self.scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        self.scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(hub_inertia)
        self.scObject.hub.sigma_BNInit = _col(sigma0_BN)
        self.scObject.hub.omega_BN_BInit = _col(omega0_BN_B)
        sim.AddModelToTask(task_name, self.scObject)

        # Earth as a point mass via the gravity factory.
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.earth = self.gravFactory.createEarth()
        self.earth.isCentralBody = True
        self.gravFactory.addBodiesTo(self.scObject)
        self.mu = self.earth.mu

        oe = orbitalMotion.ClassicElements()
        oe.a = orbit_a
        oe.e = orbit_e
        oe.i = orbit_i
        oe.Omega = orbit_Omega
        oe.omega = orbit_omega
        oe.f = orbit_f
        rN, vN = orbitalMotion.elem2rv(self.mu, oe)
        self.scObject.hub.r_CN_NInit = rN
        self.scObject.hub.v_CN_NInit = vN

        # IMU: identity mounting, zero lever arm, constant bias plus modest white noise.
        self.imuObject = imuSensor.ImuSensor()
        self.imuObject.ModelTag = "imuSensor"
        self.imuObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.imuObject.sensorPos_B = [[0.0], [0.0], [0.0]]
        self.imuObject.senRotBias = _col(gyro_bias)
        self.imuObject.senTransBias = _col(accel_bias)
        self.imuObject.PMatrixGyro = (gyro_noise_psd * np.eye(3)).tolist()
        self.imuObject.PMatrixAccel = (accel_noise_psd * np.eye(3)).tolist()
        sim.AddModelToTask(task_name, self.imuObject)


def hub_initial_state(scObject: spacecraft.Spacecraft) -> dict:
    """Reads a spacecraft's live hub initial-condition attributes — `sigma_BNInit`,
    `r_CN_NInit`, `v_CN_NInit`, `omega_BN_BInit` — into a dict keyed `sigma0_BN`, `r0_N`,
    `v0_N`, `omega0_BN_B`, picking up whatever a Monte Carlo run dispersed onto the hub before
    this runs."""
    return {
        "sigma0_BN": np.array(scObject.hub.sigma_BNInit, dtype=float).flatten(),
        "r0_N": np.array(scObject.hub.r_CN_NInit, dtype=float).flatten(),
        "v0_N": np.array(scObject.hub.v_CN_NInit, dtype=float).flatten(),
        "omega0_BN_B": np.array(scObject.hub.omega_BN_BInit, dtype=float).flatten(),
    }
