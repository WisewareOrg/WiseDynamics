"""The pytest plugin, exercised through real pytest sessions in subprocesses, loaded by its
`pytest11` entry point."""

import re
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

README = Path(__file__).parent.parent / "wise_dynamics" / "basilisk" / "README.md"

SCENARIO_TEST = """
import numpy as np
from Basilisk.fswAlgorithms import inertial3D


class Guidance(inertial3D.inertial3D):
    period_s = 1.0


def test_case(sim_case_index, sim_duration_s, sim_scenario, sim_run_file):
    datasets = sim_scenario.run(Guidance(), {}, duration_s=sim_duration_s)
    path = sim_run_file(
        datasets,
        {
            "probe_noise_seed": sim_scenario.vehicle.imu.RNGSeed,
            "probe_gyro_bias_P_rad_s": np.array(
                sim_scenario.vehicle.imu.senRotBias, dtype=float
            ).flatten(),
        },
    )
    assert path.name == f"run_{sim_case_index:04d}.h5"
    assert path.is_file()
"""

SHORT = ("--sim-duration-s", "3")

# Case 5 of seed 0: run files recorded with this seeding reproduce only while these hold.
CASE_5_NOISE_SEED = 2153058428
CASE_5_DRAWS = {
    "sigma0_BN": [0.10274604175442446, 0.19692436106593408, -0.3223839260439085],
    "omega0_BN_B_rad_s": [0.012184953200484915, 0.01815985986877613, 0.014939997512415095],
    "gyro_bias_P_rad_s": [8.079773541256839e-05, -0.00011052054402547108, -5.069458483787486e-05],
    "accel_bias_P_m_s2": [0.00012226550330897274, 1.6712413497495933e-05, 9.917025892881453e-05],
}


@pytest.fixture
def scenario_dir(pytester: pytest.Pytester) -> pytest.Pytester:
    """A pytester directory holding a one-test scenario."""
    pytester.makepyfile(test_scenario=SCENARIO_TEST)
    return pytester


def _run_session(scenario_dir: pytest.Pytester, out: str, *args: str, passed: int) -> list[dict]:
    """Run the scenario's session into `out` and return each run file's attributes and datasets."""
    out_dir = scenario_dir.path / out
    scenario_dir.runpytest_subprocess("--sim-out", str(out_dir), *SHORT, *args).assert_outcomes(
        passed=passed
    )
    runs = []
    for path in sorted(out_dir.glob("run_*.h5")):
        with h5py.File(path, "r") as run_file:
            runs.append(
                {
                    "name": path.name,
                    "attrs": {key: run_file.attrs[key] for key in run_file.attrs},
                    "t_s": run_file["t_s"][:],
                    "sigma_BN": run_file["truth/sigma_BN"][:],
                    "omega_BN_B": run_file["truth/omega_BN_B"][:],
                }
            )
    return runs


def _draws_of(run: dict) -> dict[str, np.ndarray]:
    return {
        key.removeprefix("draw_"): np.asarray(value)
        for key, value in run["attrs"].items()
        if key.startswith("draw_")
    }


_DISPERSION = (
    ("sigma0_BN", Leo500.sigma0_BN, 0.01),
    ("omega0_BN_B_rad_s", Leo500.omega0_BN_B_rad_s, 0.2 * np.pi / 180.0),
    ("gyro_bias_P_rad_s", ImuSpacecraft.gyro_bias_P_rad_s, 1e-4),
    ("accel_bias_P_m_s2", ImuSpacecraft.accel_bias_P_m_s2, 1e-4),
)
"""Leo500's dispersion, name, nominal and per-axis 1-sigma, in draw order, as its docstring
states."""


def _case_streams(seed: int, index: int) -> list[np.random.SeedSequence]:
    """Case `index`'s two streams, derived as the README states."""
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)


def _standard_draws(seed: int, index: int) -> dict[str, np.ndarray]:
    """The standard draws of case `index`, derived as the README states."""
    draws_stream, _ = _case_streams(seed, index)
    rng = np.random.default_rng(draws_stream)
    return {
        name: np.asarray(nominal) + rng.normal(0.0, std, size=3)
        for name, nominal, std in _DISPERSION
    }


def test_case_seeding_matches_recorded_runs(scenario_dir):
    (run,) = _run_session(
        scenario_dir, "out", "--sim-runs", "852", "test_scenario.py::test_case[5]", passed=1
    )
    assert run["name"] == "run_0005.h5"
    assert run["attrs"]["probe_noise_seed"] == CASE_5_NOISE_SEED
    assert {name: value.tolist() for name, value in _draws_of(run).items()} == CASE_5_DRAWS


def test_a_case_is_the_same_whatever_the_run_count(scenario_dir):
    (small,) = _run_session(
        scenario_dir, "small", "--sim-runs", "8", "test_scenario.py::test_case[5]", passed=1
    )
    (large,) = _run_session(
        scenario_dir, "large", "--sim-runs", "852", "test_scenario.py::test_case[5]", passed=1
    )
    assert small["attrs"]["probe_noise_seed"] == large["attrs"]["probe_noise_seed"]
    for name, value in _draws_of(small).items():
        assert np.array_equal(value, _draws_of(large)[name])


def test_each_case_records_the_standard_draws_its_vehicle_used(scenario_dir):
    runs = _run_session(scenario_dir, "out", "--sim-runs", "3", passed=3)

    assert [run["attrs"]["case"] for run in runs] == [0, 1, 2]
    for index, run in enumerate(runs):
        draws = _draws_of(run)
        expected = _standard_draws(0, index)
        assert draws.keys() == expected.keys()
        for name, value in expected.items():
            assert np.array_equal(draws[name], value), (index, name)
        assert np.array_equal(run["attrs"]["probe_gyro_bias_P_rad_s"], draws["gyro_bias_P_rad_s"])
        assert np.array_equal(run["sigma_BN"][0], draws["sigma0_BN"])
        assert np.allclose(run["omega_BN_B"][0], draws["omega0_BN_B_rad_s"], rtol=0, atol=1e-15)
    for name in _draws_of(runs[0]):
        assert not np.allclose(_draws_of(runs[0])[name], _draws_of(runs[1])[name])


def test_the_run_file_carries_the_case_root_attrs(scenario_dir):
    runs = _run_session(scenario_dir, "out", "--sim-runs", "2", passed=2)

    run = runs[1]
    attrs = run["attrs"]
    assert run["name"] == "run_0001.h5"
    assert attrs["case"] == 1
    assert attrs["seed"] == 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", attrs["created_utc"])
    assert attrs["versions_bsk"] == version("bsk")
    assert attrs["versions_wise_dynamics"] == version("wise-dynamics")
    probes = {"probe_noise_seed", "probe_gyro_bias_P_rad_s"}
    draws = {f"draw_{name}" for name, _, _ in _DISPERSION}
    assert (
        set(attrs)
        == {
            "case",
            "seed",
            "created_utc",
            "versions_bsk",
            "versions_wise_dynamics",
        }
        | draws
        | probes
    )


def test_attrs_cannot_replace_the_run_files_own(pytester):
    pytester.makepyfile(
        """
        import numpy as np
        import pytest

        def test_case(sim_run_file):
            with pytest.raises(ValueError, match=r"own attrs \\['seed'\\]"):
                sim_run_file({"t_s": (np.zeros(1), "s")}, {"seed": 1})
        """
    )
    out_dir = pytester.path / "out"
    pytester.runpytest_subprocess("--sim-runs", "1", "--sim-out", str(out_dir)).assert_outcomes(
        passed=1
    )
    assert not out_dir.exists()


def test_the_scenario_option_selects_a_registered_scenario(scenario_dir):
    """A second scenario, registered by the consumer's conftest, is the scenario the session
    builds when `--sim-scenario` names it: its attitude is dispersed about its own nominal."""
    scenario_dir.makeconftest(
        """
        from wise_dynamics.basilisk import SCENARIOS, Leo500

        class Flipped(Leo500):
            sigma0_BN = (-0.1, -0.2, 0.3)

        SCENARIOS["flipped"] = Flipped
        """
    )
    (run,) = _run_session(
        scenario_dir, "out", "--sim-runs", "1", "--sim-scenario", "flipped", passed=1
    )
    assert np.allclose(run["sigma_BN"][0], [-0.1, -0.2, 0.3], rtol=0, atol=0.1)


def test_workers_share_the_default_out_dir(scenario_dir, monkeypatch):
    """`TMPDIR` points the default directory into this test's own directory, where it can be
    found."""
    monkeypatch.setenv("TMPDIR", str(scenario_dir.path))
    result = scenario_dir.runpytest_subprocess("-n", "2", "--sim-runs", "2", *SHORT)

    result.assert_outcomes(passed=2)
    files = list(scenario_dir.path.rglob("run_*.h5"))
    assert len(files) == 2
    assert len({path.parent for path in files}) == 1
    result.stdout.fnmatch_lines([f"wise-dynamics: 2 run files in {files[0].parent}"])


def test_workers_write_to_the_relative_out_dir_the_controller_resolved(scenario_dir):
    """Each worker changes directory before the plugin configures it, and `--sim-out` is
    relative; the run files land in the one directory the controller resolved."""
    scenario_dir.makeconftest(
        """
        import os
        import pytest

        @pytest.hookimpl(tryfirst=True)
        def pytest_configure(config):
            if hasattr(config, "workerinput"):
                worker_dir = config.rootpath / config.workerinput["workerid"]
                worker_dir.mkdir()
                os.chdir(worker_dir)
        """
    )
    scenario_dir.runpytest_subprocess(
        "-n", "2", "--sim-runs", "2", "--sim-out", "out", *SHORT
    ).assert_outcomes(passed=2)

    in_out_dir = sorted((scenario_dir.path / "out").glob("run_*.h5"))
    assert len(in_out_dir) == 2
    assert sorted(scenario_dir.path.rglob("run_*.h5")) == in_out_dir


def test_each_case_runs_for_sim_duration_s(scenario_dir):
    (run,) = _run_session(
        scenario_dir, "out", "--sim-runs", "1", "--sim-duration-s", "20", passed=1
    )
    assert run["t_s"].tolist() == [0.0, 10.0, 20.0]


def test_a_session_runs_without_xdist(scenario_dir):
    runs = _run_session(scenario_dir, "out", "-p", "no:xdist", "--sim-runs", "1", passed=1)
    assert len(runs) == 1


def test_a_session_with_no_case_creates_nothing_in_the_temporary_directory(
    pytester, tmp_path, monkeypatch
):
    """The default output directory sits under `TMPDIR`; a session with no case leaves `TMPDIR`
    as it found it."""
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    pytester.makepyfile("def test_nothing():\n    pass\n")
    before = sorted(temporary.iterdir())
    pytester.runpytest_subprocess(f"--basetemp={tmp_path / 'basetemp'}").assert_outcomes(passed=1)
    assert sorted(temporary.iterdir()) == before


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--sim-runs", "0", "--sim-runs must be at least 1, got 0"),
        ("--sim-duration-s", "0", "--sim-duration-s must be positive, got 0.0"),
    ],
)
def test_out_of_range_options_are_usage_errors(pytester, option, value, message):
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess(option, value)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"ERROR: wise-dynamics: {message}"])


def test_an_unknown_scenario_fails_collection_naming_it_and_the_known_ones(pytester):
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess("--sim-scenario", "bogus")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(
        [
            "*UsageError: wise-dynamics: --sim-scenario: unknown scenario 'bogus'; "
            "known scenarios: leo500"
        ]
    )


def test_an_out_dir_holding_run_files_is_a_usage_error(pytester):
    out_dir = pytester.path / "out"
    out_dir.mkdir()
    (out_dir / "run_0000.h5").write_bytes(b"")
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess("--sim-out", str(out_dir))
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"ERROR: --sim-out {out_dir} already holds 1 run file;*"])


def test_a_session_using_no_sim_fixture_imports_no_basilisk(pytester):
    """The plugin loads into every session of an environment with `wise-dynamics` installed."""
    pytester.makepyfile(
        """
        import sys

        def test_modules():
            assert "wise_dynamics.pytest_plugin" in sys.modules
            assert [name for name in sys.modules if name.startswith("Basilisk")] == []
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_readme_scenario_runs_as_a_session(pytester):
    (example,) = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    pytester.makepyfile(test_example=example)
    out_dir = pytester.path / "out"
    pytester.runpytest_subprocess(
        "--sim-runs", "2", "--sim-duration-s", "20", "--sim-out", str(out_dir)
    ).assert_outcomes(passed=2)

    files = sorted(out_dir.glob("run_*.h5"))
    assert len(files) == 2
    for path in files:
        with h5py.File(path, "r") as run_file:
            assert run_file["t_s"][:].tolist() == [0.0, 10.0, 20.0]
            assert run_file["guidance/sigma_RN"].attrs["units"] == "MRP"
            assert run_file.attrs["versions_guidance"] == "1.0"
