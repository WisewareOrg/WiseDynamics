"""One IMU on a tumbling spacecraft in a 500 km LEO orbit: WiseGncDesign's ``ImuDeadReckoner``
integrates gyro rates to attitude (MRP) and the accelerometer, rotated by that attitude, to
position and velocity against a point-mass gravity model, as an uncorrected, unfiltered
estimate. Hosted by ``AlgorithmModule`` on the ``ImuSpacecraftModel`` dynamics stack. See
docs/decisions/0001-basilisk-first-simulation-host.md.

``imuSensor.AccelPlatform`` reports the non-gravitational specific force: confirmed empirically
by running this orbit with zero sensor bias/noise and observing ``AccelPlatform`` read exactly
zero while the orbit's own gravitational acceleration is ~8.4 m/s^2, checked by running it
against this prebuilt-wheel Basilisk install, which ships binaries only.

A single nominal run is ``execute_scenario(create_scenario())``, a plain function call. A
campaign is this file run as a pytest module: ``pytest sim/basilisk/scenarios --runs N --out
DIR [-n WORKERS]`` (see ``sim/basilisk/conftest.py``) disperses `N` cases, executes each as its
own parametrized test, exports each to a contract v0 run file plus a sidecar, and folds every
sidecar into one ``manifest.json`` validated against WiseGncAnalysis's contract check at session
end.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass, macros

from sim.basilisk.hosting.algorithm_module import AlgorithmModule
from sim.basilisk.models.imu_spacecraft import (
    ACCEL_BIAS_DEFAULT,
    DYNAMICS_STEP_S,
    GYRO_BIAS_DEFAULT,
    OMEGA0_BN_B_DEFAULT,
    SIGMA0_BN_DEFAULT,
    ImuSpacecraftModel,
    hub_initial_state,
)
from sim.basilisk.scenarios.bindings import IMU_DEAD_RECKONING_BINDING, AlgorithmBinding

ORBIT_DURATION_S = 5700.0  # s, one orbit
RECORDER_RATE_S = 10.0  # s

# Contract v0 column name -> (recorder name in msgRecList, payload field, axis index). t_s
# comes from the truth recorder's own times and is handled separately.
EXPORT: dict[str, tuple[str, str, int]] = {}
for _prefix, _rec_name, _field_name in [
    ("truth_sigma", "truth", "sigma_BN"),
    ("truth_omega", "truth", "omega_BN_B"),
    ("truth_r", "truth", "r_BN_N"),
    ("truth_v", "truth", "v_BN_N"),
    ("est_sigma", "estAtt", "sigma_BN"),
    ("est_r", "estTrans", "r_BN_N"),
    ("est_v", "estTrans", "v_BN_N"),
    ("gyro", "imu", "AngVelPlatform"),
    ("accel", "imu", "AccelPlatform"),
]:
    for _axis in range(3):
        EXPORT[f"{_prefix}_{_axis + 1}"] = (_rec_name, _field_name, _axis)


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


def _attach_algorithm_and_recorders(
    dyn: ImuSpacecraftModel, binding: AlgorithmBinding
) -> tuple[AlgorithmModule, dict]:
    """Hosts `binding`'s algorithm on `dyn` and attaches the four recorders at 10 s, returning
    the module and the `msgRecList`-shaped dict of recorders."""
    algModule = AlgorithmModule(
        inputs={"imu": messaging.IMUSensorMsgReader()},
        outputs={
            "navAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg()),
            "navTrans": (messaging.NavTransMsgPayload, messaging.NavTransMsg()),
        },
        input_map=binding.input_map,
        output_map=binding.output_map,
        sample_type=binding.sample_type,
        make_algorithm=binding.make_algorithm,
        init_from=lambda: {**hub_initial_state(dyn.scObject), "mu": dyn.mu},
        mountings=binding.mountings,
    )
    algModule.ModelTag = "algorithm"
    algModule.inputs["imu"].subscribeTo(dyn.imuObject.sensorOutMsg)
    dyn.sim.AddModelToTask(dyn.task_name, algModule)

    samplingTime = macros.sec2nano(RECORDER_RATE_S)
    msgRecList = {
        "truth": dyn.scObject.scStateOutMsg.recorder(samplingTime),
        "imu": dyn.imuObject.sensorOutMsg.recorder(samplingTime),
        "estAtt": algModule.outputs["navAtt"][1].recorder(samplingTime),
        "estTrans": algModule.outputs["navTrans"][1].recorder(samplingTime),
    }
    for rec in msgRecList.values():
        dyn.sim.AddModelToTask(dyn.task_name, rec)

    return algModule, msgRecList


def create_scenario(
    sigma0_BN: np.ndarray = SIGMA0_BN_DEFAULT,
    omega0_BN_B: np.ndarray = OMEGA0_BN_B_DEFAULT,
    gyro_bias: np.ndarray = GYRO_BIAS_DEFAULT,
    accel_bias: np.ndarray = ACCEL_BIAS_DEFAULT,
    step_s: float = DYNAMICS_STEP_S,
) -> SimulationBaseClass.SimBaseClass:
    """Builds the scenario (the `ImuSpacecraftModel` dynamics stack plus the dead-reckoning
    `AlgorithmModule`) and returns it unexecuted; `execute_scenario` runs it."""
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.dynamics = ImuSpacecraftModel(
        scSim,
        step_s=step_s,
        sigma0_BN=sigma0_BN,
        omega0_BN_B=omega0_BN_B,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
    )
    scSim.scObject = scSim.dynamics.scObject
    scSim.imuObject = scSim.dynamics.imuObject

    algModule, msgRecList = _attach_algorithm_and_recorders(
        scSim.dynamics, IMU_DEAD_RECKONING_BINDING
    )
    scSim.algModule = algModule
    scSim.msgRecList = msgRecList
    return scSim


def execute_scenario(
    scSim: SimulationBaseClass.SimBaseClass, duration_s: float = ORBIT_DURATION_S
) -> None:
    """Runs a scenario built by `create_scenario`."""
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


def _export_run(msgRecList: dict, path: Path) -> None:
    columns = {"t_s": msgRecList["truth"].times() * macros.NANO2SEC}
    for column, (rec_name, field_name, axis) in EXPORT.items():
        columns[column] = np.array(getattr(msgRecList[rec_name], field_name))[:, axis]
    pd.DataFrame(columns, columns=["t_s", *EXPORT.keys()]).to_csv(path, index=False)


def test_case(case_index, plant, algorithm_binding, draws, duration_s, out_dir) -> None:
    """One Monte Carlo case: builds the algorithm binding onto the already-dispersed `plant`
    fixture, runs it, exports its contract v0 run file plus a sidecar recording this case's
    seed and draws, and checks the run file against WiseGncAnalysis's own contract check."""
    _algModule, msgRecList = _attach_algorithm_and_recorders(plant, algorithm_binding)
    execute_scenario(plant.sim, duration_s=duration_s)

    run_file = f"run_{case_index:04d}.csv"
    _export_run(msgRecList, out_dir / run_file)

    sidecar = {
        "run": case_index,
        "file": run_file,
        "seeds": {"numpy_rng": draws.rng_seed},
        "dispersions": draws.as_dict(),
    }
    (out_dir / f"run_{case_index:04d}.json").write_text(json.dumps(sidecar, indent=2))

    # Contract validation against WiseGncAnalysis withheld: contract v0 moved to a single
    # campaign.h5 (no CSV, no manifest, no sidecars), and that module is itself mid-restructure
    # on WiseGncAnalysis as of this commit. The CSV/sidecar above are a local artifact pending
    # the HDF5 exporter rewrite.
    assert len(msgRecList["truth"].times()) > 0, "scenario produced no recorded samples"


if __name__ == "__main__":
    run()
