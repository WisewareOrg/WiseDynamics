"""One IMU on a tumbling spacecraft in a 500 km LEO orbit: WiseGncDesign's ``ImuDeadReckoner``
integrates gyro rates to attitude (MRP) and the accelerometer, rotated by that attitude, to
position and velocity against a point-mass gravity model, as an uncorrected, unfiltered
estimate. Hosted by ``AlgorithmModule`` on the ``ImuSpacecraftModel`` dynamics stack. See
docs/decisions/0001-basilisk-first-simulation-host.md.

``imuSensor.AccelPlatform`` reports the non-gravitational specific force: confirmed empirically
(2026-10-31) by running this orbit with zero sensor bias/noise and observing ``AccelPlatform``
read exactly zero while the orbit's own gravitational acceleration is ~8.4 m/s^2, checked by
running it against this prebuilt-wheel Basilisk install, which ships binaries only.
"""

from dataclasses import dataclass

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass, macros
from gnc_algorithms.imu_dead_reckoning import ImuDeadReckoner, ImuSample

from sim.basilisk.hosting.algorithm_module import AlgorithmModule, FieldMap
from sim.basilisk.models.imu_spacecraft import (
    ACCEL_BIAS_DEFAULT,
    DYNAMICS_STEP_S,
    GYRO_BIAS_DEFAULT,
    OMEGA0_BN_B_DEFAULT,
    SIGMA0_BN_DEFAULT,
    ImuSpacecraftModel,
    hub_initial_state,
)

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


def create_scenario(
    sigma0_BN: np.ndarray = SIGMA0_BN_DEFAULT,
    omega0_BN_B: np.ndarray = OMEGA0_BN_B_DEFAULT,
    gyro_bias: np.ndarray = GYRO_BIAS_DEFAULT,
    accel_bias: np.ndarray = ACCEL_BIAS_DEFAULT,
    step_s: float = DYNAMICS_STEP_S,
) -> SimulationBaseClass.SimBaseClass:
    """Builds the scenario (the `ImuSpacecraftModel` dynamics stack plus a dead-reckoning
    `AlgorithmModule`) and returns it unexecuted; `execute_scenario` runs it. Every object
    dispersions or retention need to reach is stored as an attribute of the returned
    `SimBaseClass`, per the Basilisk Monte Carlo `Controller` convention: dotted-path
    dispersions address them there, and the reference keeps the C++-wrapped objects alive for
    the sim's lifetime."""
    scSim = SimulationBaseClass.SimBaseClass()

    scSim.dynamics = ImuSpacecraftModel(
        scSim,
        step_s=step_s,
        sigma0_BN=sigma0_BN,
        omega0_BN_B=omega0_BN_B,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
    )
    dyn = scSim.dynamics
    # Dispersion dotted paths ('scObject.hub.sigma_BNInit', 'imuObject.senRotBias', ...)
    # address these directly on scSim, per the Monte Carlo Controller convention.
    scSim.scObject = dyn.scObject
    scSim.imuObject = dyn.imuObject

    # Dead-reckoning estimator: identity mounting (the IMU's own dcm_PB), Earth's mu shared with
    # truth so estimate and truth integrate against the same gravity constant. Initial state is
    # read from the live hub at Reset() time (hub_initial_state reads it fresh each call), so a
    # Monte Carlo run's estimate starts aligned with that run's actual, dispersed truth.
    scSim.algModule = AlgorithmModule(
        inputs={"imu": messaging.IMUSensorMsgReader()},
        outputs={
            "navAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg()),
            "navTrans": (messaging.NavTransMsgPayload, messaging.NavTransMsg()),
        },
        input_map=[
            FieldMap("imu.AngVelPlatform", "omega_B", frame="P->B"),
            FieldMap("imu.AccelPlatform", "accel_B", frame="P->B"),
        ],
        output_map=[
            FieldMap("sigma_BN", "navAtt.sigma_BN"),
            FieldMap("r_N", "navTrans.r_BN_N"),
            FieldMap("v_N", "navTrans.v_BN_N"),
        ],
        sample_type=ImuSample,
        make_algorithm=lambda t0, init: ImuDeadReckoner(
            init["sigma0_BN"], init["r0_N"], init["v0_N"], t0, dyn.mu
        ),
        init_from=lambda: hub_initial_state(dyn.scObject),
        mountings={"imu": np.array(dyn.imuObject.dcm_PB)},
    )
    scSim.algModule.ModelTag = "deadReckoning"
    scSim.algModule.inputs["imu"].subscribeTo(dyn.imuObject.sensorOutMsg)
    scSim.AddModelToTask(dyn.task_name, scSim.algModule)

    # Recorders at 10 s: truth, IMU output, both estimate messages.
    samplingTime = macros.sec2nano(RECORDER_RATE_S)
    scSim.msgRecList = {
        "truth": dyn.scObject.scStateOutMsg.recorder(samplingTime),
        "imu": dyn.imuObject.sensorOutMsg.recorder(samplingTime),
        "estAtt": scSim.algModule.outputs["navAtt"][1].recorder(samplingTime),
        "estTrans": scSim.algModule.outputs["navTrans"][1].recorder(samplingTime),
    }
    for rec in scSim.msgRecList.values():
        scSim.AddModelToTask(dyn.task_name, rec)

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
    is accepted for signature parity with other Basilisk scenario scripts."""
    del show_plots
    scSim = create_scenario(sigma0_BN=sigma0_BN, omega0_BN_B=omega0_BN_B, step_s=step_s)
    execute_scenario(scSim, duration_s=duration_s)
    return _collect_result(scSim)


if __name__ == "__main__":
    run()
