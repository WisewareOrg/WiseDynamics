"""852-run Monte Carlo campaign over `sim.basilisk.scenarios.imu_dead_reckoning`, exported to the
WiseGncAnalysis contract v0 (`run_<NNNN>.csv` + `manifest.json`).

Dispersed per run: initial MRP, initial body rate, gyro bias, accelerometer bias; all RNG seeds.
The spec names `MRPDispersionPerAxis` for the initial MRP "or equivalent" with no stated
magnitude; `MRPDispersionPerAxis` sets each axis to an independent, nominal-value-ignoring
absolute draw (its own bounds, default a full `[0, 2*pi]` per axis), which does not fit
perturbing around a nonzero nominal attitude. `NormalVectorCartDispersion` is used instead -- the
same class Basilisk's own shipped `scenarioMonteCarloAttRW.py` uses for exactly this purpose (a
normal perturbation of an attitude MRP around its nominal value) -- with a standard deviation of
0.01 per axis, chosen to be small relative to the ~0.1-0.3 nominal components (the same
small-fraction-of-nominal proportion as that shipped example's own MRP dispersion). This magnitude
is a judgment call, not something the spec states; flagged for review.

`manifest.json` carries a top-level `versions` key (bsk, gnc-algorithms, wise-dynamics) tracing
the campaign back to the code that produced it; WiseGncAnalysis's contract check at commit
fcb4c38 ignores unknown top-level manifest keys, confirmed by running it against a manifest
carrying this key.
"""

import argparse
import ast
import json
import multiprocessing
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distribution
from importlib.metadata import version as pkg_version
from pathlib import Path

import pandas as pd
from Basilisk.utilities import macros
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy
from Basilisk.utilities.MonteCarlo.Dispersions import NormalVectorCartDispersion

from sim.basilisk.models.imu_spacecraft import (
    ACCEL_BIAS_DEFAULT,
    GYRO_BIAS_DEFAULT,
    OMEGA0_BN_B_DEFAULT,
    SIGMA0_BN_DEFAULT,
)
from sim.basilisk.scenarios.imu_dead_reckoning import create_scenario, execute_scenario

DEFAULT_RUNS = 852

MRP_STD = 0.01  # -, per axis; see module docstring
RATE_STD = 0.2 * macros.D2R  # rad/s, per axis
GYRO_BIAS_STD = 1e-4  # rad/s, per axis
ACCEL_BIAS_STD = 1e-4  # m/s^2, per axis

DISP_MRP = "scObject.hub.sigma_BNInit"
DISP_RATE = "scObject.hub.omega_BN_BInit"
DISP_GYRO_BIAS = "imuObject.senRotBias"
DISP_ACCEL_BIAS = "imuObject.senTransBias"

# Column order of the contract v0 run file (contract/README.md in WiseGncAnalysis).
_VECTOR_COLUMNS = [
    ("truth_sigma", "truth.sigma_BN"),
    ("truth_omega", "truth.omega_BN_B"),
    ("truth_r", "truth.r_BN_N"),
    ("truth_v", "truth.v_BN_N"),
    ("est_sigma", "estAtt.sigma_BN"),
    ("est_r", "estTrans.r_BN_N"),
    ("est_v", "estTrans.v_BN_N"),
    ("gyro", "imu.AngVelPlatform"),
    ("accel", "imu.AccelPlatform"),
]
CSV_COLUMNS = ["t_s"] + [f"{prefix}_{axis}" for prefix, _ in _VECTOR_COLUMNS for axis in (1, 2, 3)]


def build_controller(
    num_runs: int, archive_dir: str, worker_count: int | None = None
) -> Controller:
    """Builds the Monte Carlo `Controller`: dispersions, seed dispersal, and retention of the
    scenario's four recorded messages at their 10 s rate. Does not execute the campaign."""
    controller = Controller()
    controller.setSimulationFunction(create_scenario)
    controller.setExecutionFunction(execute_scenario)
    controller.setExecutionCount(num_runs)
    controller.setThreadCount(
        worker_count if worker_count is not None else multiprocessing.cpu_count()
    )
    controller.setShouldDisperseSeeds(True)
    controller.setArchiveDir(archive_dir)

    controller.addDispersion(
        NormalVectorCartDispersion(DISP_MRP, SIGMA0_BN_DEFAULT.tolist(), [MRP_STD] * 3)
    )
    controller.addDispersion(
        NormalVectorCartDispersion(DISP_RATE, OMEGA0_BN_B_DEFAULT.tolist(), [RATE_STD] * 3)
    )
    controller.addDispersion(
        NormalVectorCartDispersion(DISP_GYRO_BIAS, GYRO_BIAS_DEFAULT.tolist(), [GYRO_BIAS_STD] * 3)
    )
    controller.addDispersion(
        NormalVectorCartDispersion(
            DISP_ACCEL_BIAS, ACCEL_BIAS_DEFAULT.tolist(), [ACCEL_BIAS_STD] * 3
        )
    )

    policy = RetentionPolicy()
    policy.addMessageLog("truth", ["sigma_BN", "omega_BN_B", "r_BN_N", "v_BN_N"])
    policy.addMessageLog("imu", ["AngVelPlatform", "AccelPlatform"])
    policy.addMessageLog("estAtt", ["sigma_BN"])
    policy.addMessageLog("estTrans", ["r_BN_N", "v_BN_N"])
    controller.addRetentionPolicy(policy)

    return controller


def _run_dataframe(messages: dict) -> pd.DataFrame:
    time_ns = messages["truth.sigma_BN"][:, 0]
    columns = {"t_s": time_ns * macros.NANO2SEC}
    for prefix, msg_key in _VECTOR_COLUMNS:
        vec = messages[msg_key][:, 1:4]
        for axis in range(3):
            columns[f"{prefix}_{axis + 1}"] = vec[:, axis]
    return pd.DataFrame(columns, columns=CSV_COLUMNS)


def _split_parameters(params: dict) -> tuple[dict, dict]:
    """Every value in a Monte Carlo `run<N>.json` is a string (seeds are `str(int)`, dispersed
    vectors are a hand-built `"[v0,v1,v2]"` string); parses each back to its real value and
    splits into the contract's `seeds` (any `*.RNGSeed` path) and `dispersions` (everything
    else)."""
    seeds: dict[str, int] = {}
    dispersions: dict[str, object] = {}
    for key, raw in params.items():
        value = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if key.endswith(".RNGSeed"):
            seeds[key] = int(value)
        else:
            dispersions[key] = value
    return seeds, dispersions


def _git_describe(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _producer_versions() -> dict[str, str]:
    """The versions that produced this campaign, for tracing a manifest back to the code that
    made it: the installed `bsk` version, and `git describe` of each sibling checkout
    (gnc-algorithms, wise-dynamics) -- a record of what ran, not a pin."""
    gnc_dir = Path(
        json.loads(distribution("gnc-algorithms").read_text("direct_url.json"))["url"].removeprefix(
            "file://"
        )
    )
    repo_root = Path(__file__).resolve().parents[3]

    return {
        "bsk": pkg_version("bsk"),
        "gnc-algorithms": _git_describe(gnc_dir),
        "wise-dynamics": _git_describe(repo_root),
    }


def export_campaign(controller: Controller, num_runs: int, out_dir: Path) -> None:
    """Post-campaign pass (parent process, serial): reads each run's retained data and archived
    parameters back off disk and writes contract v0's `run_<NNNN>.csv` + `manifest.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs = []
    for i in range(num_runs):
        retained = controller.getRetainedData(i)
        params = controller.getParameters(i)

        run_file = f"run_{i:04d}.csv"
        _run_dataframe(retained["messages"]).to_csv(out_dir / run_file, index=False)

        seeds, dispersions = _split_parameters(params)
        manifest_runs.append(
            {"run": i, "file": run_file, "seeds": seeds, "dispersions": dispersions}
        )

    manifest = {
        "contract_version": 0,
        "producer": "WiseDynamics sim/basilisk/campaigns/imu_dead_reckoning_mc.py",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": _producer_versions(),
        "runs": manifest_runs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = str(out_dir / "_basilisk_archive")

    controller = build_controller(args.runs, archive_dir)
    failures = controller.executeSimulations()
    export_campaign(controller, args.runs, out_dir)

    print(f"{args.runs} runs, {len(failures)} failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
