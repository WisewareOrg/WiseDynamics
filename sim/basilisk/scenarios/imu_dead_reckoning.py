"""One IMU on a tumbling spacecraft in a 500 km LEO orbit: gyro rates integrated to attitude
(MRP) and the accelerometer, rotated by that attitude, integrated to position/velocity against a
point-mass gravity model, by WiseGncDesign's ``ImuDeadReckoner`` (dead reckoning only, no
correction). See docs/decisions/0001-basilisk-first-simulation-host.md.

``imuSensor.AccelPlatform`` reports non-gravitational specific force only: confirmed empirically
(2026-10-31) by running this orbit with zero sensor bias/noise and observing ``AccelPlatform``
read exactly zero while the orbit's own gravitational acceleration is ~8.4 m/s^2. No C++ source
is bundled in this prebuilt-wheel Basilisk install, so this was checked by running it rather than
by reading the module source directly.
"""

from dataclasses import dataclass

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import imuSensor, spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simHelpers,
    simIncludeGravBody,
)
from gnc_algorithms.imu_dead_reckoning import ImuDeadReckoner, ImuSample

# Orbit: 500 km circular LEO at 45 deg inclination (classical elements, Earth point-mass gravity).
ORBIT_A = 6871.0e3  # m
ORBIT_E = 0.0001  # -
ORBIT_I = 45.0 * macros.D2R  # rad
ORBIT_OMEGA_CAP = 48.2 * macros.D2R  # rad, right ascension of ascending node
ORBIT_OMEGA_LOW = 347.8 * macros.D2R  # rad, argument of periapsis
ORBIT_F = 85.3 * macros.D2R  # rad, true anomaly

# Rigid hub.
HUB_MASS = 750.0  # kg
HUB_INERTIA = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]  # kg*m^2, diagonal

# Tumble defaults.
SIGMA0_BN_DEFAULT = np.array([0.1, 0.2, -0.3])  # -, MRP
OMEGA0_BN_B_DEFAULT = np.array([1.0, 1.0, 1.0]) * macros.D2R  # rad/s per axis

# IMU error model defaults (applied to the identity-mounted sensor; modest white noise on top of
# the constant bias, zero correlation across axes).
GYRO_BIAS_DEFAULT = np.array([1e-4, -2e-4, 5e-5])  # rad/s
ACCEL_BIAS_DEFAULT = np.array([1e-4, -5e-5, 2e-4])  # m/s^2
GYRO_NOISE_PSD_DEFAULT = 1e-12  # (rad/s)^2 * s, diagonal PMatrixGyro
ACCEL_NOISE_PSD_DEFAULT = 1e-10  # (m/s^2)^2 * s, diagonal PMatrixAccel

DYNAMICS_STEP_S = 1.0  # s, dynamics/sensor/algorithm step
ORBIT_DURATION_S = 5700.0  # s, one orbit
RECORDER_RATE_S = 10.0  # s


@dataclass
class ScenarioResult:
    """Recorded arrays from one `run()` call. `time_s` is shared by every array's first axis."""

    time_s: np.ndarray
    truth_sigma_BN: np.ndarray
    truth_omega_BN_B: np.ndarray
    truth_r_BN_N: np.ndarray
    truth_v_BN_N: np.ndarray
    est_sigma_BN: np.ndarray
    est_r_BN_N: np.ndarray
    est_v_BN_N: np.ndarray
    gyro_B: np.ndarray
    accel_B: np.ndarray


class DeadReckoningModule(sysModel.SysModel):
    """Basilisk Python module wrapping `ImuDeadReckoner`: reads the IMU output message (platform
    frame), maps platform to body with the sensor's mounting DCM, steps the algorithm, and writes
    the resulting attitude/position/velocity estimate as a `NavAttMsgPayload` /
    `NavTransMsgPayload`. Initial state is read from the live spacecraft hub at `Reset()` time (not
    captured at construction) so a Monte Carlo run that disperses the hub's initial condition
    starts the reckoner aligned with that run's actual, dispersed truth."""

    def __init__(self, scObject: spacecraft.Spacecraft, dcm_PB: np.ndarray, mu: float) -> None:
        super().__init__()
        self.scObject = scObject
        self.dcm_PB = np.array(dcm_PB, dtype=float)
        self.mu = mu
        self._reckoner: ImuDeadReckoner | None = None
        self.gyroInMsg = messaging.IMUSensorMsgReader()
        self.navAttOutMsg = messaging.NavAttMsg()
        self.navTransOutMsg = messaging.NavTransMsg()

    def Reset(self, CurrentSimNanos: int) -> None:
        t0 = CurrentSimNanos * macros.NANO2SEC
        sigma0_BN = np.array(self.scObject.hub.sigma_BNInit, dtype=float).flatten()
        r0_N = np.array(self.scObject.hub.r_CN_NInit, dtype=float).flatten()
        v0_N = np.array(self.scObject.hub.v_CN_NInit, dtype=float).flatten()
        self._reckoner = ImuDeadReckoner(sigma0_BN, r0_N, v0_N, t0, self.mu)

    def UpdateState(self, CurrentSimNanos: int) -> None:
        buf = self.gyroInMsg()
        t = CurrentSimNanos * macros.NANO2SEC
        omega_P = np.array(buf.AngVelPlatform, dtype=float)
        accel_P = np.array(buf.AccelPlatform, dtype=float)
        omega_B = self.dcm_PB.T @ omega_P
        accel_B = self.dcm_PB.T @ accel_P

        state = self._reckoner.step(ImuSample(t=t, omega_B=omega_B, accel_B=accel_B))

        attOut = messaging.NavAttMsgPayload()
        attOut.sigma_BN = state.sigma_BN.tolist()
        self.navAttOutMsg.write(attOut, CurrentSimNanos, self.moduleID)

        transOut = messaging.NavTransMsgPayload()
        transOut.r_BN_N = state.r_N.tolist()
        transOut.v_BN_N = state.v_N.tolist()
        self.navTransOutMsg.write(transOut, CurrentSimNanos, self.moduleID)


def create_scenario(
    sigma0_BN: np.ndarray = SIGMA0_BN_DEFAULT,
    omega0_BN_B: np.ndarray = OMEGA0_BN_B_DEFAULT,
    gyro_bias: np.ndarray = GYRO_BIAS_DEFAULT,
    accel_bias: np.ndarray = ACCEL_BIAS_DEFAULT,
    step_s: float = DYNAMICS_STEP_S,
) -> SimulationBaseClass.SimBaseClass:
    """Builds the scenario (spacecraft, gravity, IMU, dead-reckoning module) but does not run it.
    Every object dispersions or retention need to reach is stored as an attribute of the returned
    `SimBaseClass`, per the Basilisk Monte Carlo `Controller` convention (both so dotted-path
    dispersions can address them and so Python does not garbage-collect C++-wrapped objects out
    from under the running sim)."""
    simTaskName = "simTask"
    simProcessName = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(step_s)
    scSim.dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Rigid hub, no effectors, no control.
    scSim.scObject = spacecraft.Spacecraft()
    scSim.scObject.ModelTag = "bsk-Sat"
    scSim.scObject.hub.mHub = HUB_MASS
    scSim.scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scSim.scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(HUB_INERTIA)
    scSim.scObject.hub.sigma_BNInit = [[float(x)] for x in sigma0_BN]
    scSim.scObject.hub.omega_BN_BInit = [[float(x)] for x in omega0_BN_B]
    scSim.AddModelToTask(simTaskName, scSim.scObject)

    # Earth as a point mass, no SPICE.
    scSim.gravFactory = simIncludeGravBody.gravBodyFactory()
    scSim.earth = scSim.gravFactory.createEarth()
    scSim.earth.isCentralBody = True
    scSim.gravFactory.addBodiesTo(scSim.scObject)
    mu = scSim.earth.mu

    oe = orbitalMotion.ClassicElements()
    oe.a = ORBIT_A
    oe.e = ORBIT_E
    oe.i = ORBIT_I
    oe.Omega = ORBIT_OMEGA_CAP
    oe.omega = ORBIT_OMEGA_LOW
    oe.f = ORBIT_F
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scSim.scObject.hub.r_CN_NInit = rN
    scSim.scObject.hub.v_CN_NInit = vN

    # IMU: identity mounting, zero lever arm, constant bias plus modest white noise.
    scSim.imuObject = imuSensor.ImuSensor()
    scSim.imuObject.ModelTag = "imuSensor"
    scSim.imuObject.scStateInMsg.subscribeTo(scSim.scObject.scStateOutMsg)
    scSim.imuObject.sensorPos_B = [[0.0], [0.0], [0.0]]
    scSim.imuObject.senRotBias = [[float(x)] for x in gyro_bias]
    scSim.imuObject.senTransBias = [[float(x)] for x in accel_bias]
    scSim.imuObject.PMatrixGyro = (GYRO_NOISE_PSD_DEFAULT * np.eye(3)).tolist()
    scSim.imuObject.PMatrixAccel = (ACCEL_NOISE_PSD_DEFAULT * np.eye(3)).tolist()
    scSim.AddModelToTask(simTaskName, scSim.imuObject)

    # Dead-reckoning estimator: identity mounting DCM (dcm_PB default), Earth's mu shared with
    # truth so estimate and truth integrate against the same gravity constant.
    scSim.drModule = DeadReckoningModule(scSim.scObject, np.array(scSim.imuObject.dcm_PB), mu)
    scSim.drModule.ModelTag = "deadReckoning"
    scSim.drModule.gyroInMsg.subscribeTo(scSim.imuObject.sensorOutMsg)
    scSim.AddModelToTask(simTaskName, scSim.drModule)

    # Recorders at 10 s: truth, IMU output, both estimate messages.
    samplingTime = macros.sec2nano(RECORDER_RATE_S)
    scSim.msgRecList = {
        "truth": scSim.scObject.scStateOutMsg.recorder(samplingTime),
        "imu": scSim.imuObject.sensorOutMsg.recorder(samplingTime),
        "estAtt": scSim.drModule.navAttOutMsg.recorder(samplingTime),
        "estTrans": scSim.drModule.navTransOutMsg.recorder(samplingTime),
    }
    for rec in scSim.msgRecList.values():
        scSim.AddModelToTask(simTaskName, rec)

    return scSim


def execute_scenario(
    scSim: SimulationBaseClass.SimBaseClass, duration_s: float = ORBIT_DURATION_S
) -> None:
    """Runs a scenario built by `create_scenario`. Split out from scenario creation so the Monte
    Carlo `Controller` can register this as its execution function."""
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(duration_s))
    scSim.ExecuteSimulation()


def _collect_result(scSim: SimulationBaseClass.SimBaseClass) -> ScenarioResult:
    rec = scSim.msgRecList
    dcm_PB = np.array(scSim.imuObject.dcm_PB)
    gyro_B = np.array(rec["imu"].AngVelPlatform) @ dcm_PB
    accel_B = np.array(rec["imu"].AccelPlatform) @ dcm_PB
    return ScenarioResult(
        time_s=rec["truth"].times() * macros.NANO2SEC,
        truth_sigma_BN=np.array(rec["truth"].sigma_BN),
        truth_omega_BN_B=np.array(rec["truth"].omega_BN_B),
        truth_r_BN_N=np.array(rec["truth"].r_BN_N),
        truth_v_BN_N=np.array(rec["truth"].v_BN_N),
        est_sigma_BN=np.array(rec["estAtt"].sigma_BN),
        est_r_BN_N=np.array(rec["estTrans"].r_BN_N),
        est_v_BN_N=np.array(rec["estTrans"].v_BN_N),
        gyro_B=gyro_B,
        accel_B=accel_B,
    )


def run(
    duration_s: float = ORBIT_DURATION_S,
    step_s: float = DYNAMICS_STEP_S,
    show_plots: bool = False,
    sigma0_BN: np.ndarray = SIGMA0_BN_DEFAULT,
    omega0_BN_B: np.ndarray = OMEGA0_BN_B_DEFAULT,
) -> ScenarioResult:
    """Builds and runs the scenario single-process, returning the recorded arrays. `show_plots`
    is accepted for parity with other Basilisk scenario scripts; this scenario draws no plots."""
    del show_plots
    scSim = create_scenario(sigma0_BN=sigma0_BN, omega0_BN_B=omega0_BN_B, step_s=step_s)
    execute_scenario(scSim, duration_s=duration_s)
    return _collect_result(scSim)


if __name__ == "__main__":
    run()
