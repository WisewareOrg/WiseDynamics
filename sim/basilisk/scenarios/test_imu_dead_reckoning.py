"""One IMU on a tumbling spacecraft in a 500 km LEO orbit: WiseGncDesign's ``ImuDeadReckoner``
integrates gyro rates to attitude (MRP) and the accelerometer, rotated by that attitude, to
position and velocity against a point-mass gravity model, as an uncorrected, unfiltered
estimate. Booted as a named flight-software instance against the ``ImuSpacecraftModel`` plant,
through the host adapter in WiseGncDesign's C&DH partition (``gnc_cdh.basilisk.make_fsw``). See
docs/decisions/0001-basilisk-first-simulation-host.md.

``imuSensor.AccelPlatform`` reports the non-gravitational specific force: confirmed empirically
by running this orbit with zero sensor bias/noise and observing ``AccelPlatform`` read exactly
zero while the orbit's own gravitational acceleration is ~8.4 m/s^2, checked by running it
against this prebuilt-wheel Basilisk install, which ships binaries only.

A single nominal run is ``execute_scenario(create_scenario())``, a plain function call. A
campaign is this file run as a pytest module: ``pytest sim/basilisk/scenarios --runs N --out
DIR [-n WORKERS]`` (see ``sim/basilisk/conftest.py``) disperses `N` cases and executes each as
its own parametrized test, writing one ``run_<NNNN>.h5`` per case directly into ``--out``, per
SPEC-demo.md §2's layout.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros
from gnc_cdh.basilisk import make_fsw

from sim.basilisk.conftest import Draws, _producer_versions
from sim.basilisk.models.imu_spacecraft import (
    ACCEL_BIAS_DEFAULT,
    DYNAMICS_STEP_S,
    GYRO_BIAS_DEFAULT,
    OMEGA0_BN_B_DEFAULT,
    SIGMA0_BN_DEFAULT,
    ImuSpacecraftModel,
)

FSW_NAME = "imu_dead_reckoning"

ORBIT_DURATION_S = 5700.0  # s, one orbit
RECORDER_RATE_S = 10.0  # s

# SPEC-demo.md §2's run file layout: dataset name -> (recorder name in msgRecList, payload
# field, units). t_s comes from the truth recorder's own times and is handled separately.
EXPORT: dict[str, tuple[str, str, str]] = {
    "truth/sigma_BN": ("truth", "sigma_BN", "MRP"),
    "truth/omega_BN_B": ("truth", "omega_BN_B", "rad/s"),
    "truth/r_N": ("truth", "r_BN_N", "m"),
    "truth/v_N": ("truth", "v_BN_N", "m/s"),
    "est/sigma_BN": ("estAtt", "sigma_BN", "MRP"),
    "est/r_N": ("estTrans", "r_BN_N", "m"),
    "est/v_N": ("estTrans", "v_BN_N", "m/s"),
    "imu/gyro_P": ("imu", "AngVelPlatform", "rad/s"),
    "imu/accel_P": ("imu", "AccelPlatform", "m/s^2"),
}


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
    dyn: ImuSpacecraftModel, fsw_name: str = FSW_NAME
) -> tuple[Any, dict]:
    """Boots `fsw_name` on `dyn` through WiseGncDesign's C&DH partition and attaches the four
    recorders at 10 s, returning the hosted module and the `msgRecList`-shaped dict of
    recorders."""
    algModule = make_fsw(
        fsw_name,
        inputs=dyn.readers(),
        initial_state=dyn.initial_state,
        mu=dyn.mu,
    )
    algModule.ModelTag = "algorithm"
    dyn.sim.AddModelToTask(dyn.task_name, algModule)

    samplingTime = macros.sec2nano(RECORDER_RATE_S)
    msgRecList = {
        "truth": dyn.scObject.scStateOutMsg.recorder(samplingTime),
        "imu": dyn.imuObject.sensorOutMsg.recorder(samplingTime),
        "estAtt": algModule.outputs["nav_att"][1].recorder(samplingTime),
        "estTrans": algModule.outputs["nav_trans"][1].recorder(samplingTime),
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
    """Builds the scenario (the `ImuSpacecraftModel` plant plus the dead-reckoning flight
    software) and returns it unexecuted; `execute_scenario` runs it."""
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

    algModule, msgRecList = _attach_algorithm_and_recorders(scSim.dynamics)
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


def _write_run(path: Path, msgRecList: dict, case_index: int, draws: Draws) -> None:
    """Writes one `run_<NNNN>.h5` per SPEC-demo.md §2: root attrs (`case`, `seed`, `producer`,
    `created_utc`, `versions_*`, one `draw_<name>` per dispersed quantity) and the ten float64
    datasets, each carrying a string `units` attribute."""
    versions = _producer_versions()
    t_s = msgRecList["truth"].times() * macros.NANO2SEC

    with h5py.File(path, "w") as f:
        f.attrs["case"] = case_index
        f.attrs["seed"] = draws.rng_seed
        f.attrs["producer"] = "WiseDynamics sim/basilisk/scenarios/test_imu_dead_reckoning.py"
        f.attrs["created_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        f.attrs["versions_bsk"] = versions["bsk"]
        f.attrs["versions_gnc_algorithms"] = versions["gnc-algorithms"]
        f.attrs["versions_wise_dynamics"] = versions["wise-dynamics"]
        for name, value in draws.as_dict().items():
            f.attrs[f"draw_{name}"] = np.asarray(value, dtype=np.float64)

        ds = f.create_dataset("t_s", data=t_s.astype(np.float64))
        ds.attrs["units"] = "s"
        for dataset_name, (rec_name, field_name, units) in EXPORT.items():
            data = np.array(getattr(msgRecList[rec_name], field_name), dtype=np.float64)
            ds = f.create_dataset(dataset_name, data=data)
            ds.attrs["units"] = units


def _assert_run_loads(path: Path) -> None:
    """Confirms WiseGncAnalysis's own loader reads this run back cleanly; `load_run` raises
    `RunDataError` naming the file and the problem otherwise."""
    from gnc_analysis.runs import load_run

    load_run(path)


def test_case(case_index, plant, fsw_name, draws, duration_s, out_dir) -> None:
    """One Monte Carlo case: boots `fsw_name` onto the already-dispersed `plant` fixture, runs
    it, writes its `run_<NNNN>.h5`, and checks that WiseGncAnalysis's own loader reads it
    back."""
    _algModule, msgRecList = _attach_algorithm_and_recorders(plant, fsw_name)
    execute_scenario(plant.sim, duration_s=duration_s)
    assert len(msgRecList["truth"].times()) > 0, "scenario produced no recorded samples"

    run_file = out_dir / f"run_{case_index:04d}.h5"
    _write_run(run_file, msgRecList, case_index, draws)

    _assert_run_loads(run_file)


if __name__ == "__main__":
    run()
