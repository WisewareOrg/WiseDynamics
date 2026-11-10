"""The pytest plugin, exercised through real pytest sessions in subprocesses.

Each session loads the plugin by its `pytest11` entry point. Each session's test file carries its
own marks in source.
"""

import json
import re
import subprocess
import textwrap
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest

from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

README = Path(__file__).parent.parent / "wise_dynamics" / "basilisk" / "README.md"
REPOSITORY = Path(__file__).parent.parent
SESSION = "s"
SERIAL = ("-p", "no:xdist")
XDIST = ("-n", "auto")

SCENARIO_HEADER = """
import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D


class Guidance(inertial3D.inertial3D):
    period_s = 1.0
"""

CASE_TEST = """
def NAME(SELFsim_case_index, sim_duration_s, sim_scenario, sim_run_file):
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

# Case 5 of seed 0: run files recorded with this seeding reproduce only while these hold.
CASE_5_NOISE_SEED = 2153058428
CASE_5_DRAWS = {
    "sigma0_BN": [0.10274604175442446, 0.19692436106593408, -0.3223839260439085],
    "omega0_BN_B_rad_s": [0.012184953200484915, 0.01815985986877613, 0.014939997512415095],
    "gyro_bias_P_rad_s": [8.079773541256839e-05, -0.00011052054402547108, -5.069458483787486e-05],
    "accel_bias_P_m_s2": [0.00012226550330897274, 1.6712413497495933e-05, 9.917025892881453e-05],
}


def _case_test(marks: tuple[str, ...] = (), method: bool = False, name: str = "test_case") -> str:
    """Write the source of one scenario test, decorated with marks.

    Args:
        marks: Each mark's call, such as `"sim_runs(3)"`, without `pytest.mark.`.
        method: Whether the test is a method of a class, taking `self`.
        name: The test function's name.

    Returns:
        The test's source.
    """
    decorators = "".join(f"@pytest.mark.{mark}\n" for mark in marks)
    body = CASE_TEST.replace("NAME", name).replace("SELF", "self, " if method else "")
    return "\n" + decorators + body.lstrip("\n")


def _source(*tests: str) -> str:
    """Join scenario tests into one test file's source.

    Args:
        *tests: Each test's source, as `_case_test` returns it.

    Returns:
        The file's source.
    """
    return SCENARIO_HEADER + "\n".join(tests)


def _point(pytester: pytest.Pytester, root: str = "monte_carlo", name: str = SESSION) -> Path:
    """Write the session pointer `<root>/SESSION` and return the session's folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        root: The `sim_out` root, relative to the rootdir.
        name: The active session's name.

    Returns:
        `<rootdir>/<root>/<name>`, which the pointer makes the session's folder.
    """
    (pytester.path / root).mkdir(parents=True, exist_ok=True)
    (pytester.path / root / "SESSION").write_text(f"{name}\n")
    return pytester.path / root / name


def _read_runs(directory: Path) -> list[dict]:
    """Read back each run file in a test's directory.

    Args:
        directory: The test's directory within the session.

    Returns:
        One mapping per run file, in name order: its `name`, root `attrs`, `t_s` (s), `sigma_BN`
        (MRP) and `omega_BN_B` (rad/s).
    """
    runs = []
    for path in sorted(directory.glob("run_*.h5")):
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


def _run_session(
    pytester: pytest.Pytester,
    *args: str,
    runs: int,
    passed: int,
    out: str = "out",
    duration_s: float = 3,
    marks: tuple[str, ...] = (),
) -> list[dict]:
    """Run the one-test scenario, marked as given, into session `s` under `out` and read it back.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        *args: Further pytest arguments.
        runs: The `sim_runs` mark's case count.
        passed: The number of tests the session must pass.
        out: The `sim_out` ini setting, a name under `pytester`; the active session is `s`.
        duration_s: The `sim_duration_s` mark's seconds.
        marks: Further marks' calls.

    Returns:
        One mapping per run file in the test's directory, as `_read_runs` returns them.
    """
    session = _point(pytester, out)
    pytester.makepyfile(
        test_scenario=_source(
            _case_test((f"sim_runs({runs})", f"sim_duration_s({duration_s})", *marks))
        )
    )
    pytester.runpytest_subprocess("-o", f"sim_out={out}", *args).assert_outcomes(passed=passed)
    return _read_runs(session / "test_scenario" / "test_case")


def _draws_of(run: dict) -> dict[str, np.ndarray]:
    """Collect a run's `draw_<name>` root attributes.

    Args:
        run: One run file's mapping, as `_run_session` returns it.

    Returns:
        Draw name, without its `draw_` prefix, to its value.
    """
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
    """Derive case `index`'s two streams, as the README states.

    Args:
        seed: The session seed.
        index: The case number.

    Returns:
        The draw stream, then the noise stream.
    """
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)


def _standard_draws(seed: int, index: int) -> dict[str, np.ndarray]:
    """Derive the standard draws of case `index`, as the README states.

    Args:
        seed: The session seed.
        index: The case number.

    Returns:
        Draw name to its value, in draw order, in the units of `_DISPERSION`.
    """
    draws_stream, _ = _case_streams(seed, index)
    rng = np.random.default_rng(draws_stream)
    return {
        name: np.asarray(nominal) + rng.normal(0.0, std, size=3)
        for name, nominal, std in _DISPERSION
    }


def test_case_seeding_matches_recorded_runs(pytester: pytest.Pytester):
    """Case 5 of seed 0 draws, and seeds its noise with, the recorded values.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (run,) = _run_session(pytester, "test_scenario.py::test_case[5]", runs=852, passed=1)
    assert run["name"] == "run_0005.h5"
    assert run["attrs"]["probe_noise_seed"] == CASE_5_NOISE_SEED
    assert {name: value.tolist() for name, value in _draws_of(run).items()} == CASE_5_DRAWS


def test_a_case_is_the_same_whatever_the_run_count(pytester: pytest.Pytester):
    """Case 5 draws the same values in a test of 8 cases as in one of 852.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (small,) = _run_session(
        pytester, "test_scenario.py::test_case[5]", runs=8, passed=1, out="small"
    )
    (large,) = _run_session(
        pytester, "test_scenario.py::test_case[5]", runs=852, passed=1, out="large"
    )
    assert small["attrs"]["probe_noise_seed"] == large["attrs"]["probe_noise_seed"]
    for name, value in _draws_of(small).items():
        assert np.array_equal(value, _draws_of(large)[name])


def test_each_case_records_the_standard_draws_its_vehicle_used(pytester: pytest.Pytester):
    """Each case records the standard draws, and its vehicle and placement use them.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    runs = _run_session(pytester, runs=3, passed=3)

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


def test_the_run_file_carries_the_case_root_attrs(pytester: pytest.Pytester):
    """A run file carries exactly the case's root attributes and the caller's probes.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    runs = _run_session(pytester, runs=2, passed=2)

    run = runs[1]
    attrs = run["attrs"]
    assert run["name"] == "run_0001.h5"
    assert attrs["case"] == 1
    assert attrs["seed"] == 0
    assert attrs["fidelity"] == "lo"
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
            "fidelity",
            "created_utc",
            "versions_bsk",
            "versions_wise_dynamics",
            "versions_wise_dynamics_install",
            "versions_wise_dynamics_commit",
            "versions_wise_dynamics_requested_revision",
            "versions_wise_dynamics_dirty",
        }
        | draws
        | probes
    )


def test_the_fidelity_mark_reaches_the_run_file(pytester: pytest.Pytester):
    """A `sim_fidelity` mark sets the fidelity the run file records.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (run,) = _run_session(pytester, runs=1, passed=1, marks=('sim_fidelity("hi")',))
    assert run["attrs"]["fidelity"] == "hi"


def test_attrs_cannot_replace_the_run_files_own(pytester: pytest.Pytester):
    """Caller attributes naming one of the file's own are refused, and no file is written.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    out_dir = pytester.path / "out"
    _point(pytester, "out")
    pytester.makepyfile(
        "import numpy as np\n"
        "import pytest\n\n\n"
        "@pytest.mark.sim_runs(1)\n"
        + textwrap.dedent(
            """
            def test_case(sim_run_file):
                with pytest.raises(ValueError, match=r"own attrs \\['seed'\\]"):
                    sim_run_file({"t_s": (np.zeros(1), "s")}, {"seed": 1})
            """
        )
    )
    pytester.runpytest_subprocess("-o", "sim_out=out").assert_outcomes(passed=1)
    assert not list(out_dir.rglob("run_*.h5"))


def _default_session(pytester: pytest.Pytester, *args: str, runs: int = 2) -> pytest.RunResult:
    """Run the one-test scenario into session `s` under the default `monte_carlo` root.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        *args: Further pytest arguments.
        runs: The `sim_runs` mark's case count.

    Returns:
        The session's result, which must have passed every case.
    """
    _point(pytester)
    pytester.makepyfile(
        test_scenario=_source(_case_test((f"sim_runs({runs})", "sim_duration_s(3)")))
    )
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=runs)
    return result


def test_a_session_with_no_sim_out_writes_to_the_default_run_folder(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Run files land in `<rootdir>/monte_carlo/<session>/<test file>/<test>/`; `TMPDIR` is empty.

    The test's directory holds the run files and `planned.json` and no stamp folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory, holding the `TMPDIR` used.
        monkeypatch: pytest's fixture for setting `TMPDIR` for this test only.
    """
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    result = _default_session(pytester, f"--basetemp={tmp_path / 'basetemp'}", *SERIAL)

    directory = pytester.path / "monte_carlo" / SESSION / "test_scenario" / "test_case"
    assert sorted(path.name for path in directory.iterdir()) == [
        "planned.json",
        "run_0000.h5",
        "run_0001.h5",
    ]
    assert len(list(pytester.path.rglob("run_*.h5"))) == 2
    result.stdout.fnmatch_lines([f"wise-dynamics: 2 run files in {directory}"])
    assert list(temporary.iterdir()) == []


def test_ci_writes_under_the_session_and_plans_the_capped_count(pytester: pytest.Pytester):
    """`--ci` caps the cases and writes into the session, not a `ci` folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(5)", "sim_duration_s(3)"))))
    pytester.runpytest_subprocess(*SERIAL, "--ci").assert_outcomes(passed=2)

    directory = pytester.path / "monte_carlo" / SESSION / "test_scenario" / "test_case"
    assert [run["name"] for run in _read_runs(directory)] == ["run_0000.h5", "run_0001.h5"]
    assert json.loads((directory / "planned.json").read_text()) == {
        "runs": 2,
        "nodeid": "test_scenario.py::test_case",
    }
    assert not (pytester.path / "monte_carlo" / "ci").exists()


def test_an_empty_sim_out_ini_setting_is_a_usage_error(pytester: pytest.Pytester):
    """An empty `sim_out` ini value stops the session with a usage error naming the setting.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\nsim_out =\n")
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))))
    result = pytester.runpytest_subprocess(*SERIAL)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "sim_out" in result.stderr.str()
    assert not list(pytester.path.rglob("run_*.h5"))


def test_workers_write_to_the_relative_out_dir_resolved_against_rootdir(
    pytester: pytest.Pytester,
):
    """A relative `sim_out` resolves against rootdir, whatever directory a worker is in.

    Each worker changes directory before the plugin configures it, and the session pointer and run
    files are still found under the rootdir's `out`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeconftest(
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
    _point(pytester, "out")
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(2)", "sim_duration_s(3)"))))
    pytester.runpytest_subprocess("-o", "sim_out=out", "-n", "2").assert_outcomes(passed=2)

    in_out_dir = sorted(
        (pytester.path / "out" / SESSION / "test_scenario" / "test_case").glob("run_*.h5")
    )
    assert len(in_out_dir) == 2
    assert sorted(pytester.path.rglob("run_*.h5")) == in_out_dir


def test_each_case_runs_for_sim_duration_s(pytester: pytest.Pytester):
    """A case runs for its `sim_duration_s` mark, recording every 10 s.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (run,) = _run_session(pytester, runs=1, passed=1, duration_s=20)
    assert run["t_s"].tolist() == [0.0, 10.0, 20.0]


def test_a_session_runs_without_xdist(pytester: pytest.Pytester):
    """A session runs with pytest-xdist disabled.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    runs = _run_session(pytester, "-p", "no:xdist", runs=1, passed=1)
    assert len(runs) == 1


@pytest.mark.parametrize("xdist", [("-p", "no:xdist"), ("-n", "2")], ids=["serial", "xdist"])
def test_two_campaigns_in_one_session_each_run_in_their_own_directory(
    pytester: pytest.Pytester, xdist: tuple[str, ...]
):
    """Two tests differing in case count are two campaigns in one session's folder.

    Each test's run files are in its own directory, nothing is in the session's root, and the
    summary reports each directory.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    session = _point(pytester)
    pytester.makepyfile(
        test_two=_source(
            _case_test(name="test_a", marks=("sim_runs(3)", "sim_duration_s(3)")),
            _case_test(name="test_b", marks=("sim_runs(2)", "sim_duration_s(3)")),
        )
    )
    result = pytester.runpytest_subprocess(*xdist)

    result.assert_outcomes(passed=5)
    dir_a, dir_b = session / "test_two" / "test_a", session / "test_two" / "test_b"
    assert [run["name"] for run in _read_runs(dir_a)] == [f"run_000{i}.h5" for i in range(3)]
    assert [run["name"] for run in _read_runs(dir_b)] == [f"run_000{i}.h5" for i in range(2)]
    assert len(list(pytester.path.rglob("run_*.h5"))) == 5
    result.stdout.fnmatch_lines(
        [f"wise-dynamics: 3 run files in {dir_a}", f"wise-dynamics: 2 run files in {dir_b}"]
    )


def test_each_session_holds_its_own_run_files(pytester: pytest.Pytester):
    """Pointing at another session between two runs leaves the first session's files in place.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))))
    first = _point(pytester, name="first")
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)
    second = _point(pytester, name="second")
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)

    for session in (first, second):
        assert [run["name"] for run in _read_runs(session / "test_scenario" / "test_case")] == [
            "run_0000.h5"
        ]


def test_test_files_outside_the_rootdir_are_named_by_their_stem(
    pytester: pytest.Pytester, tmp_path_factory: pytest.TempPathFactory
):
    """A test file outside `--rootdir` is named by its stem, so same-named tests do not collide.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path_factory: pytest's factory for directories outside the rootdir.
    """
    out = pytester.path / "out"
    session = _point(pytester, "out")
    source = _source(_case_test(("sim_runs(1)", "sim_duration_s(3)")))
    files = []
    for stem in ("test_one", "test_two"):
        files.append(tmp_path_factory.mktemp(stem) / f"{stem}.py")
        files[-1].write_text(source)
    result = pytester.runpytest_subprocess(
        "-p", "no:xdist", "--rootdir", str(pytester.path), "-o", "sim_out=out", *map(str, files)
    )

    result.assert_outcomes(passed=2)
    for stem in ("test_one", "test_two"):
        assert [run["name"] for run in _read_runs(session / stem / "test_case")] == ["run_0000.h5"]
    assert len(list(out.rglob("run_*.h5"))) == 2
    assert not [path for file in files for path in file.parent.rglob("run_*.h5")]


def test_a_class_and_a_nested_class_qualify_the_directory(pytester: pytest.Pytester):
    """A test's directory is its file path, each class outermost first, then its function.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    out_dir = _point(pytester)
    inner = textwrap.indent(_case_test(name="test_inner", method=True), "        ")
    outer = textwrap.indent(_case_test(name="test_outer", method=True), "    ")
    source = (
        SCENARIO_HEADER
        + "\npytestmark = [pytest.mark.sim_runs(1), pytest.mark.sim_duration_s(3)]\n\n\n"
        + f"class TestOuter:{outer}\n    class TestInner:{inner}"
    )
    (pytester.path / "tests" / "sim").mkdir(parents=True)
    (pytester.path / "tests" / "sim" / "test_nested.py").write_text(source)
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    base = out_dir / "tests" / "sim" / "test_nested"
    found = sorted(path.relative_to(out_dir).parent for path in out_dir.rglob("run_*.h5"))
    assert found == [
        base.relative_to(out_dir) / "TestOuter" / "TestInner" / "test_inner",
        base.relative_to(out_dir) / "TestOuter" / "test_outer",
    ]


def test_the_closest_mark_wins_over_class_and_module_marks(pytester: pytest.Pytester):
    """A mark on a function overrides the class's, which overrides the module's.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makepyfile(
        """
        import pytest

        pytestmark = [pytest.mark.sim_runs(2), pytest.mark.sim_duration_s(11)]


        def test_module(sim_case_index, sim_duration_s):
            assert sim_duration_s == 11


        @pytest.mark.sim_duration_s(12)
        class TestGroup:
            def test_class(self, sim_case_index, sim_duration_s):
                assert sim_duration_s == 12

            @pytest.mark.sim_duration_s(13)
            @pytest.mark.sim_runs(3)
            def test_function(self, sim_case_index, sim_duration_s):
                assert sim_duration_s == 13
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=7)


def test_a_session_with_no_case_creates_no_run_folder(pytester: pytest.Pytester):
    """A session that runs no case creates neither the default run folder nor any run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makepyfile("def test_nothing():\n    pass\n")
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)
    assert not (pytester.path / "monte_carlo").exists()


def test_an_unmarked_test_gets_the_defaults(pytester: pytest.Pytester):
    """An unmarked test flies `leo500` at `lo` fidelity for 5700 s, in 8 cases.

    No case runs the vehicle: the values come from the fixtures.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makepyfile(
        """
        from wise_dynamics.basilisk.scenarios import SCENARIOS


        def test_defaults(sim_case_index, sim_duration_s, sim_scenario):
            assert sim_duration_s == 5700
            assert type(sim_scenario) is SCENARIOS["leo500"]
            assert sim_scenario.fidelity == "lo"
        """
    )
    pytester.runpytest_subprocess("-p", "no:xdist").assert_outcomes(passed=8)


def _bad_mark_session(pytester: pytest.Pytester, mark: str, *args: str) -> pytest.RunResult:
    """Run a serial session whose one test carries `mark`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        mark: The mark's call, without `pytest.mark.`.
        *args: Further pytest arguments.

    Returns:
        The session's result, which must have collected nothing.
    """
    _point(pytester)
    pytester.makepyfile(
        f"import pytest\n\n\n@pytest.mark.{mark}\ndef test_case(sim_case_index):\n    pass\n"
    )
    result = pytester.runpytest_subprocess("-p", "no:xdist", *args)
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    return result


@pytest.mark.parametrize(
    ("mark", "name"),
    [
        ('sim_scenario("bogus")', "sim_scenario"),
        ("sim_runs(0)", "sim_runs"),
        ("sim_runs(True)", "sim_runs"),
        ("sim_duration_s(0)", "sim_duration_s"),
        ('sim_fidelity("ultra")', "sim_fidelity"),
        ("sim_runs()", "sim_runs"),
        ("sim_runs(1, 2)", "sim_runs"),
        ("sim_runs(3, n=3)", "sim_runs"),
    ],
)
def test_a_bad_mark_fails_collection_naming_the_test_and_the_mark(
    pytester: pytest.Pytester, mark: str, name: str
):
    """A bad mark fails collection, naming the test's node id and the mark.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        mark: The bad mark's call, without `pytest.mark.`.
        name: The mark's name.
    """
    result = _bad_mark_session(pytester, mark)
    result.stdout.fnmatch_lines([f"*.py::test_case: {name}: *"])
    if name == "sim_scenario":
        result.stdout.fnmatch_lines(["*unknown scenario 'bogus'; known scenarios: leo500*"])
    assert not list(pytester.path.rglob("run_*.h5"))


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        (False, {"test_a": 5, "test_b": 1, "test_c": 8}),
        (True, {"test_a": 2, "test_b": 1, "test_c": 2}),
    ],
    ids=["without-ci", "ci"],
)
def test_ci_caps_every_tests_case_count_at_two(
    pytester: pytest.Pytester, ci: bool, expected: dict[str, int]
):
    """`--ci` caps a test's cases at 2; a test asking for fewer runs its own.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        ci: Whether the session passes `--ci`.
        expected: Test name to the number of cases it collects.
    """
    _point(pytester)
    pytester.makepyfile(
        test_cap="""
        import pytest

        @pytest.mark.sim_runs(5)
        def test_a(sim_case_index):
            pass

        @pytest.mark.sim_runs(1)
        def test_b(sim_case_index):
            pass

        def test_c(sim_case_index):
            pass
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:xdist", "--collect-only", "-q", *["--ci"] * ci)
    assert result.ret == pytest.ExitCode.OK
    for name, count in expected.items():
        assert (
            len([line for line in result.outlines if line.startswith(f"test_cap.py::{name}[")])
            == count
        )


def test_ci_leaves_fidelity_and_duration_to_the_marks(pytester: pytest.Pytester):
    """`--ci` changes the case count only.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.sim_runs(5)
        @pytest.mark.sim_fidelity("hi")
        @pytest.mark.sim_duration_s(7)
        def test_case(sim_case_index, sim_duration_s, sim_scenario):
            assert sim_duration_s == 7
            assert sim_scenario.fidelity == "hi"
        """
    )
    pytester.runpytest_subprocess("-p", "no:xdist", "--ci").assert_outcomes(passed=2)


def test_ci_still_rejects_a_bad_mark(pytester: pytest.Pytester):
    """A `sim_runs(0)` mark fails collection under `--ci`; the cap does not repair it.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    result = _bad_mark_session(pytester, "sim_runs(0)", "--ci")
    result.stdout.fnmatch_lines(["*.py::test_case: sim_runs: *"])


def test_each_removed_option_is_unknown(pytester: pytest.Pytester):
    """Each `--sim-*` option is an unrecognized argument.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    options = {
        "--sim-scenario": "leo500",
        "--sim-runs": "2",
        "--sim-fidelity": "lo",
        "--sim-duration-s": "3",
        "--sim-out": "out",
    }
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess(*(arg for item in options.items() for arg in item))
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    unrecognized = [line for line in result.stderr.lines if "unrecognized arguments:" in line]
    assert len(unrecognized) == 1
    for option in options:
        assert option in unrecognized[0]


def test_the_four_marks_are_registered_and_sim_out_is_not(pytester: pytest.Pytester):
    """`--strict-markers` accepts the four marks and rejects `sim_out`, which no longer exists.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    result = _strict_session(pytester, ['sim_out("out")'])
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*'sim_out' not found in `markers` configuration option*"])

    marks = ['sim_scenario("leo500")', "sim_runs(1)", 'sim_fidelity("lo")', "sim_duration_s(3)"]
    _strict_session(pytester, marks).assert_outcomes(passed=1)


def _strict_session(pytester: pytest.Pytester, marks: list[str]) -> pytest.RunResult:
    """Run a `--strict-markers` session whose one test carries `marks`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        marks: Each mark's call, without `pytest.mark.`.

    Returns:
        The session's result.
    """
    decorators = "".join(f"@pytest.mark.{mark}\n" for mark in marks)
    _point(pytester)
    pytester.makepyfile(
        f"import pytest\n\n\n{decorators}def test_case(sim_case_index):\n    pass\n"
    )
    return pytester.runpytest_subprocess("-p", "no:xdist", "--strict-markers")


def test_a_rerun_of_other_cases_keeps_the_planned_count_and_the_other_files(
    pytester: pytest.Pytester,
):
    """Running one case at a time into a session leaves `planned.json` at `sim_runs`.

    Case 0 runs under pytest-xdist, then case 1 alone by its node id; the second run neither
    rewrites case 0's file nor plans only the one case it selected.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester)
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(2)", "sim_duration_s(3)"))))
    pytester.runpytest_subprocess(*XDIST, "test_scenario.py::test_case[0]").assert_outcomes(
        passed=1
    )
    directory = session / "test_scenario" / "test_case"
    assert sorted(path.name for path in directory.iterdir()) == ["planned.json", "run_0000.h5"]
    assert json.loads((directory / "planned.json").read_text()) == {
        "runs": 2,
        "nodeid": "test_scenario.py::test_case",
    }
    before = (directory / "run_0000.h5").stat().st_mtime_ns

    pytester.runpytest_subprocess(*SERIAL, "test_scenario.py::test_case[1]").assert_outcomes(
        passed=1
    )

    assert sorted(path.name for path in directory.iterdir()) == [
        "planned.json",
        "run_0000.h5",
        "run_0001.h5",
    ]
    assert json.loads((directory / "planned.json").read_text()) == {
        "runs": 2,
        "nodeid": "test_scenario.py::test_case",
    }
    assert (directory / "run_0000.h5").stat().st_mtime_ns == before
    assert [path.name for path in session.iterdir()] == ["test_scenario"]


def test_a_collect_only_session_creates_nothing(pytester: pytest.Pytester):
    """`--collect-only` creates no run folder and no `planned.json`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester)
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(2)", "sim_duration_s(3)"))))
    result = pytester.runpytest_subprocess(*SERIAL, "--collect-only", "-q")
    assert result.ret == pytest.ExitCode.OK
    assert not session.exists()
    assert list((pytester.path / "monte_carlo").iterdir()) == [
        pytester.path / "monte_carlo" / "SESSION"
    ]


def _git(directory: Path, *args: str) -> str:
    """Run `git` in `directory`.

    Args:
        directory: The repository.
        *args: The `git` arguments.

    Returns:
        The command's standard output, without its trailing newline.
    """
    identity = ("-c", "user.name=Test", "-c", "user.email=test@example.com")
    result = subprocess.run(
        ["git", *identity, "-c", "commit.gpgsign=false", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_repository(directory: Path, dirty: bool = False) -> str:
    """Create a git repository with one commit, and optionally an uncommitted file.

    Args:
        directory: The repository to create.
        dirty: Whether to leave an untracked file in it.

    Returns:
        The commit's 40-hex id.
    """
    directory.mkdir()
    _git(directory, "init", "-q")
    (directory / "tracked.txt").write_text("tracked")
    _git(directory, "add", "tracked.txt")
    _git(directory, "commit", "-q", "-m", "first")
    if dirty:
        (directory / "untracked.txt").write_text("untracked")
    return _git(directory, "rev-parse", "HEAD")


def _provenance_session(
    pytester: pytest.Pytester, direct_url: dict | None, *args: str
) -> pytest.RunResult:
    """Run the one-case scenario with the install's `direct_url.json` replaced by `direct_url`.

    The plugin's `distribution` lookup returns a stand-in whose `read_text("direct_url.json")` is
    the JSON, or `None` for an install that wrote none.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        direct_url: The install's `direct_url.json` content, or `None` for none.
        *args: Further pytest arguments.

    Returns:
        The session's result.
    """
    pytester.makeconftest(
        "import json\n\n"
        "import wise_dynamics.pytest_plugin as plugin\n\n\n"
        "class Distribution:\n"
        "    def read_text(self, name):\n"
        '        assert name == "direct_url.json"\n'
        f"        return json.dumps({direct_url!r}) if {direct_url!r} is not None else None\n\n\n"
        "plugin.distribution = lambda name: Distribution()\n"
    )
    _point(pytester)
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))))
    return pytester.runpytest_subprocess(*args)


def _provenance_attrs(pytester: pytest.Pytester) -> dict:
    """Read the run file's four provenance attributes from the default run folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.

    Returns:
        `install`, `commit`, `requested_revision` and `dirty`.
    """
    prefix = "versions_wise_dynamics_"
    (run,) = _read_runs(pytester.path / "monte_carlo" / SESSION / "test_scenario" / "test_case")
    return {
        key: run["attrs"][prefix + key]
        for key in ("install", "commit", "requested_revision", "dirty")
    }


def test_a_pinned_install_records_its_commit_as_pinned(pytester: pytest.Pytester):
    """A VCS install records `pinned`, its commit, its requested revision and a clean tree.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    direct_url = {
        "url": "https://github.com/WisewareOrg/WiseDynamics",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "746add8156aa61701971cf4cdde30f092277d962",
            "requested_revision": "746add8",
        },
    }
    _provenance_session(pytester, direct_url, *SERIAL).assert_outcomes(passed=1)

    found = _provenance_attrs(pytester)
    assert found["install"] == "pinned"
    assert found["commit"] == direct_url["vcs_info"]["commit_id"]
    assert found["requested_revision"] == direct_url["vcs_info"]["requested_revision"]
    assert found["dirty"] == np.False_


@pytest.mark.parametrize("dirty", [False, True], ids=["clean", "dirty"])
def test_an_editable_install_records_its_checkouts_commit_and_dirt(
    pytester: pytest.Pytester, tmp_path: Path, dirty: bool
):
    """An editable install records `local`, the checkout's HEAD and whether it has changes.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory, holding the checkout.
        dirty: Whether the checkout has an uncommitted file.
    """
    repository = tmp_path / "checkout"
    commit = _make_repository(repository, dirty=dirty)
    direct_url = {"url": repository.as_uri(), "dir_info": {"editable": True}}
    _provenance_session(pytester, direct_url, *SERIAL).assert_outcomes(passed=1)

    found = _provenance_attrs(pytester)
    assert found["install"] == "local"
    assert found["commit"] == commit
    assert found["requested_revision"] == ""
    assert type(found["dirty"]) is np.bool_
    assert bool(found["dirty"]) is dirty


def test_wise_dynamics_own_editable_install_records_local(pytester: pytest.Pytester):
    """With nothing patched, this repository's editable install records `local` and its HEAD.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))))
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)

    found = _provenance_attrs(pytester)
    assert found["install"] == "local"
    assert found["commit"] == _git(REPOSITORY, "rev-parse", "HEAD")


@pytest.mark.parametrize("case", ["no-direct-url", "git-unavailable"])
def test_an_install_that_cannot_be_described_stops_the_session(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
):
    """An install the plugin cannot describe exits 4, names both supported installs, runs no case.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory.
        monkeypatch: pytest's fixture for setting the session's environment for this test only.
        case: `no-direct-url`, an install with no `direct_url.json`, or `git-unavailable`, an
            editable install whose checkout cannot be read because `git` fails.
    """
    direct_url = None
    if case == "git-unavailable":
        (tmp_path / "no_git").mkdir()
        monkeypatch.setenv("PATH", str(tmp_path / "no_git"))
        direct_url = {"url": REPOSITORY.as_uri(), "dir_info": {"editable": True}}
    result = _provenance_session(pytester, direct_url, *SERIAL)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    output = result.stdout.lines + result.stderr.lines
    assert len([line for line in output if "uv sync" in line]) == 1
    assert len([line for line in output if "uv pip install -e" in line]) == 1
    assert not list(pytester.path.rglob("run_*.h5"))


def test_the_plugin_runs_without_the_notebook_extra(pytester: pytest.Pytester):
    """With IPython and jupytext unimportable, a sim test still runs; the notebook module says why.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester)
    pytester.makeconftest(
        """
        import sys

        sys.modules["IPython"] = None
        sys.modules["jupytext"] = None
        """
    )
    pytester.makepyfile(
        test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))),
        test_notebook_import="""
        import re

        import pytest


        def test_the_notebook_module_names_the_extra():
            with pytest.raises(ImportError, match=re.escape("wise-dynamics[notebook]")):
                import wise_dynamics.notebook
        """,
    )
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=2)
    assert len(list((pytester.path / "monte_carlo" / SESSION).rglob("run_*.h5"))) == 1


def test_a_session_using_no_sim_fixture_imports_no_basilisk(pytester: pytest.Pytester):
    """The plugin loads into every session without importing Basilisk.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makepyfile(
        """
        import sys

        def test_modules():
            assert "wise_dynamics.pytest_plugin" in sys.modules
            assert [name for name in sys.modules if name.startswith("Basilisk")] == []
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_readme_scenario_runs_as_a_session(pytester: pytest.Pytester):
    """The README's worked example, marks included, runs in the active session and writes its files.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (example,) = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    session = _point(pytester)
    pytester.makepyfile(test_example=example)
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    files = sorted((session / "test_example" / "test_case").glob("run_*.h5"))
    assert len(files) == 2
    for path in files:
        with h5py.File(path, "r") as run_file:
            assert run_file["t_s"][:].tolist() == [0.0, 10.0, 20.0]
            assert run_file["guidance/sigma_RN"].attrs["units"] == "MRP"
            assert run_file.attrs["versions_guidance"] == "1.0"


NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"
PLAIN_AND_MONTE_CARLO = """
import pytest


@pytest.mark.sim_runs(3)
def test_monte_carlo_a(sim_case_index):
    pass


def test_monte_carlo_b(sim_case_index):
    pass


def test_plain():
    pass
"""


def test_sim_session_with_a_name_writes_the_pointer_and_collects_nothing(
    pytester: pytest.Pytester,
):
    """`--sim-session x` writes `<root>/SESSION`, prints `session: x`, exits 0 and runs no test.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makepyfile("def test_boom():\n    assert False\n")
    result = pytester.runpytest_subprocess("--sim-session", "alpha")

    assert result.ret == pytest.ExitCode.OK
    assert "session: alpha" in result.stdout.lines
    assert (pytester.path / "monte_carlo" / "SESSION").read_text() == "alpha\n"
    assert not any("test_boom" in line or "passed" in line for line in result.outlines)


def test_sim_session_writes_the_pointer_under_the_ini_root(pytester: pytest.Pytester):
    """The pointer is `<sim_out>/SESSION`, the root created if it does not exist.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\nsim_out = ini_out/deeper\n")
    result = pytester.runpytest_subprocess("--sim-session", "alpha")

    assert result.ret == pytest.ExitCode.OK
    assert (pytester.path / "ini_out" / "deeper" / "SESSION").read_text() == "alpha\n"
    assert not (pytester.path / "monte_carlo").exists()


@pytest.mark.parametrize("name", ["a", "7", "v1.2", "run_3-b", "A.b_c-d"])
def test_sim_session_accepts_a_valid_name(pytester: pytest.Pytester, name: str):
    """A name of letters, digits, `.`, `_` and `-`, starting with a letter or digit, is written.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: The session name.
    """
    result = pytester.runpytest_subprocess(f"--sim-session={name}")

    assert result.ret == pytest.ExitCode.OK
    assert (pytester.path / "monte_carlo" / "SESSION").read_text() == f"{name}\n"


@pytest.mark.parametrize("name", ["..", ".", "-x", ".hidden", "SESSION", "a/b", "_a"])
def test_sim_session_refuses_an_invalid_name_and_leaves_the_pointer(
    pytester: pytest.Pytester, name: str
):
    """An invalid name exits non-zero, not as an unknown option; the pointer is left as it was.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: The refused session name.
    """
    _point(pytester, name="keep")
    result = pytester.runpytest_subprocess(f"--sim-session={name}")

    assert result.ret != pytest.ExitCode.OK
    assert "unrecognized arguments" not in result.stderr.str() + result.stdout.str()
    assert (pytester.path / "monte_carlo" / "SESSION").read_text() == "keep\n"
    assert not (pytester.path / "SESSION").exists()


def test_sim_session_without_a_name_shows_the_active_session_and_the_others(
    pytester: pytest.Pytester,
):
    """With no name it prints `active: <name>`, then each other directory under the root.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester, name="beta")
    for other in ("alpha", "gamma"):
        (pytester.path / "monte_carlo" / other).mkdir()
    result = pytester.runpytest_subprocess("--sim-session")

    assert result.ret == pytest.ExitCode.OK
    lines = [line.strip() for line in result.stdout.lines if line.strip()]
    assert lines[0] == "active: beta"
    assert sorted(lines[1:]) == ["alpha", "gamma"]


def test_sim_session_without_a_name_and_no_pointer_says_none(pytester: pytest.Pytester):
    """With no pointer, and no root at all, it prints `active: none` and exits 0.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    result = pytester.runpytest_subprocess("--sim-session")

    assert result.ret == pytest.ExitCode.OK
    assert [line.strip() for line in result.stdout.lines if line.strip()] == ["active: none"]
    assert not (pytester.path / "monte_carlo").exists()


def test_sim_session_under_xdist_addopts_starts_no_workers(pytester: pytest.Pytester):
    """With `-n auto` in `addopts`, `--sim-session x` runs no pytest session at all.

    A conftest writes a marker file in `pytest_sessionstart`; a plain run proves the marker works,
    and the `--sim-session` run, started afterwards, leaves it absent: no controller session, so
    no worker.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\naddopts = -n auto\n")
    pytester.makeconftest(
        """
        from pathlib import Path


        def pytest_sessionstart(session):
            Path(session.config.rootpath / "started").write_text("started")
        """
    )
    pytester.makepyfile("def test_plain():\n    pass\n")
    pytester.runpytest_subprocess().assert_outcomes(passed=1)
    marker = pytester.path / "started"
    assert marker.exists()
    marker.unlink()

    result = pytester.runpytest_subprocess("--sim-session", "x")

    assert result.ret == pytest.ExitCode.OK
    assert (pytester.path / "monte_carlo" / "SESSION").read_text() == "x\n"
    assert not marker.exists()


@pytest.mark.parametrize("blocker", ["broken-pointer", "undescribable-install"])
def test_sim_session_is_not_blocked_by_a_broken_pointer_or_install(
    pytester: pytest.Pytester, blocker: str
):
    """The option needs neither a readable pointer nor a describable install.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        blocker: `broken-pointer`, an invalid name in `SESSION`, or `undescribable-install`, a
            `wise-dynamics` install with no `direct_url.json`.
    """
    if blocker == "broken-pointer":
        _point(pytester, name="../bad")
    else:
        pytester.makeconftest(
            "import wise_dynamics.pytest_plugin as plugin\n\n\n"
            "class Distribution:\n"
            "    def read_text(self, name):\n"
            "        return None\n\n\n"
            "plugin.distribution = lambda name: Distribution()\n"
        )
    result = pytester.runpytest_subprocess("--sim-session", "x")

    assert result.ret == pytest.ExitCode.OK
    assert (pytester.path / "monte_carlo" / "SESSION").read_text() == "x\n"


@pytest.mark.parametrize("xdist", [("-p", "no:xdist"), ("-n", "2")], ids=["serial", "xdist"])
def test_with_no_pointer_each_monte_carlo_test_errors_and_other_tests_run(
    pytester: pytest.Pytester, xdist: tuple[str, ...]
):
    """Each Monte Carlo test shows one `no-session` error naming `--sim-session`; plain tests pass.

    Nothing is written.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    pytester.makepyfile(test_mixed=PLAIN_AND_MONTE_CARLO)

    listing = pytester.runpytest_subprocess(*SERIAL, "--collect-only", "-q")
    assert sorted(line for line in listing.outlines if "::" in line) == [
        "test_mixed.py::test_monte_carlo_a[no-session]",
        "test_mixed.py::test_monte_carlo_b[no-session]",
        "test_mixed.py::test_plain",
    ]

    result = pytester.runpytest_subprocess(*xdist, "-rE")

    result.assert_outcomes(passed=1, errors=2)
    errored = sorted(line.split()[1] for line in result.outlines if line.startswith("ERROR "))
    assert errored == [
        "test_mixed.py::test_monte_carlo_a[no-session]",
        "test_mixed.py::test_monte_carlo_b[no-session]",
    ]
    assert result.stdout.str().count(NO_SESSION) >= 2
    assert not (pytester.path / "monte_carlo").exists()


def test_a_session_switch_after_start_does_not_split_a_run(pytester: pytest.Pytester):
    """Workers use the session the controller handed them, whatever the pointer says later.

    Each case rewrites the pointer to `other` before it writes its run file; every file still lands
    in the session the run started in, and `other` is never created.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester)
    pointer = pytester.path / "monte_carlo" / "SESSION"
    pytester.makepyfile(
        test_switch=f"""
        from pathlib import Path

        import numpy as np
        import pytest


        @pytest.mark.sim_runs(6)
        def test_case(sim_case_index, sim_run_file):
            Path({str(pointer)!r}).write_text("other\\n")
            sim_run_file({{"t_s": (np.zeros(2), "s")}})
        """
    )
    pytester.runpytest_subprocess("-n", "1").assert_outcomes(passed=6)

    assert pointer.read_text() == "other\n"
    directory = session / "test_switch" / "test_case"
    assert [path.name for path in sorted(directory.glob("run_*.h5"))] == [
        f"run_000{i}.h5" for i in range(6)
    ]
    assert not (pytester.path / "monte_carlo" / "other").exists()
