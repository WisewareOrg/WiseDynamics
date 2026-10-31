"""Options and fixtures for running `sim/basilisk/scenarios` as a Monte Carlo campaign:
`pytest sim/basilisk/scenarios --runs N --out DIR [-n WORKERS]` disperses `N` cases through
numpy's own RNG (no Basilisk `MonteCarlo.Controller`), each executed as an ordinary pytest test
case that writes its own run file directly into `--out`."""

import subprocess
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pytest
from Basilisk.utilities import SimulationBaseClass, macros

from sim.basilisk.models.imu_spacecraft import (
    ACCEL_BIAS_DEFAULT,
    GYRO_BIAS_DEFAULT,
    OMEGA0_BN_B_DEFAULT,
    SIGMA0_BN_DEFAULT,
    ImuSpacecraftModel,
)
from sim.basilisk.scenarios.bindings import IMU_DEAD_RECKONING_BINDING

# Dispersion widths, 1-sigma per axis.
MRP_STD = 0.01  # -
RATE_STD = 0.2 * macros.D2R  # rad/s
GYRO_BIAS_STD = 1e-4  # rad/s
ACCEL_BIAS_STD = 1e-4  # m/s^2


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runs", action="store", type=int, default=8)
    parser.addoption("--out", action="store", type=str, default=None)
    parser.addoption("--seed", action="store", type=int, default=0)
    parser.addoption("--duration-s", action="store", type=float, default=5700.0)


def pytest_configure(config: pytest.Config) -> None:
    out = config.getoption("--out")
    out_dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="wise_mc_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config._mc_out_dir = out_dir


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_index" in metafunc.fixturenames:
        runs = metafunc.config.getoption("--runs")
        metafunc.parametrize("case_index", range(runs))


@pytest.fixture
def out_dir(request: pytest.FixtureRequest) -> Path:
    return request.config._mc_out_dir


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--seed")


@pytest.fixture
def duration_s(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--duration-s")


@dataclass
class Draws:
    """Per-case dispersed values, drawn from a case-indexed RNG, so a case's draws reproduce
    from its own seed alone."""

    rng_seed: int
    sigma0_BN: np.ndarray
    omega0_BN_B: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray

    def as_dict(self) -> dict:
        return {
            "sigma0_BN": self.sigma0_BN.tolist(),
            "omega0_BN_B": self.omega0_BN_B.tolist(),
            "gyro_bias": self.gyro_bias.tolist(),
            "accel_bias": self.accel_bias.tolist(),
        }


@pytest.fixture
def draws(case_index: int, seed: int) -> Draws:
    rng_seed = seed + case_index
    rng = np.random.default_rng(rng_seed)
    return Draws(
        rng_seed=rng_seed,
        sigma0_BN=SIGMA0_BN_DEFAULT + rng.normal(0.0, MRP_STD, size=3),
        omega0_BN_B=OMEGA0_BN_B_DEFAULT + rng.normal(0.0, RATE_STD, size=3),
        gyro_bias=GYRO_BIAS_DEFAULT + rng.normal(0.0, GYRO_BIAS_STD, size=3),
        accel_bias=ACCEL_BIAS_DEFAULT + rng.normal(0.0, ACCEL_BIAS_STD, size=3),
    )


@pytest.fixture
def plant(draws: Draws) -> ImuSpacecraftModel:
    """Parametrizable later by vehicle; for now, the one IMU spacecraft, with this case's
    dispersed initial conditions and biases."""
    sim = SimulationBaseClass.SimBaseClass()
    return ImuSpacecraftModel(
        sim,
        sigma0_BN=draws.sigma0_BN,
        omega0_BN_B=draws.omega0_BN_B,
        gyro_bias=draws.gyro_bias,
        accel_bias=draws.accel_bias,
    )


@pytest.fixture
def algorithm_binding():
    return IMU_DEAD_RECKONING_BINDING


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
    """The versions that produced a run, for tracing it back to the code that made it: the
    installed `bsk` version, and `git describe` of each sibling checkout."""
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "bsk": pkg_version("bsk"),
        "gnc-algorithms": _git_describe(repo_root.parent / "WiseGncDesign"),
        "wise-dynamics": _git_describe(repo_root),
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Validates the whole campaign directory with WiseGncAnalysis's own loader. Runs once, on
    the controlling process, not on every pytest-xdist worker."""
    config = session.config
    if hasattr(config, "workerinput"):
        return

    out_dir = config._mc_out_dir
    if not any(out_dir.glob("run_*.h5")):
        return

    from gnc_analysis.runs import RunDataError, load_campaign

    try:
        load_campaign(out_dir)
    except RunDataError as exc:
        session.exitstatus = 1
        print(f"run export failed WiseGncAnalysis's loader: {exc}")
