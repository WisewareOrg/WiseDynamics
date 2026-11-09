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
DATA = Path(__file__).parent / "data"
SERIAL = ("-p", "no:xdist")
XDIST = ("-n", "auto")

# Each way a test names its run folder: a `sim_out` mark, the default, or the `sim_out` ini setting.
ROOT_NAMES = {"mark": "out", "default": "monte_carlo", "ini": "ini_out"}

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


def _out_mark(path: Path) -> str:
    """Say a `sim_out` mark for `path`.

    Args:
        path: The output directory.

    Returns:
        The mark's call, without `pytest.mark.`.
    """
    return f"sim_out({str(path)!r})"


def _read_runs(directory: Path) -> list[dict]:
    """Read back each run file in `directory`.

    Args:
        directory: The test directory holding the run files.

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
    """Run the one-test scenario, marked as given, into `out` and read back each run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        *args: Further pytest arguments.
        runs: The `sim_runs` mark's case count.
        passed: The number of tests the session must pass.
        out: The `sim_out` mark's directory, a name under `pytester`.
        duration_s: The `sim_duration_s` mark's seconds.
        marks: Further marks' calls.

    Returns:
        One mapping per run file in the test's directory, as `_read_runs` returns them.
    """
    out_dir = pytester.path / out
    pytester.makepyfile(
        test_scenario=_source(
            _case_test(
                (f"sim_runs({runs})", _out_mark(out_dir), f"sim_duration_s({duration_s})", *marks),
            )
        )
    )
    pytester.runpytest_subprocess(*args).assert_outcomes(passed=passed)
    return _read_runs(out_dir / "test_scenario" / "test_case")


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
    pytester.makepyfile(
        "import numpy as np\n"
        "import pytest\n\n\n"
        f"@pytest.mark.sim_runs(1)\n@pytest.mark.{_out_mark(out_dir)}\n"
        + textwrap.dedent(
            """
            def test_case(sim_run_file):
                with pytest.raises(ValueError, match=r"own attrs \\['seed'\\]"):
                    sim_run_file({"t_s": (np.zeros(1), "s")}, {"seed": 1})
            """
        )
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)
    assert not out_dir.exists()


def _default_session(
    pytester: pytest.Pytester, *args: str, ini: str | None = None, runs: int = 2
) -> pytest.RunResult:
    """Run the one-test scenario with no `sim_out` mark, under an optional `sim_out` ini setting.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        *args: Further pytest arguments.
        ini: The ini file's `sim_out` value, or `None` for no ini file.
        runs: The `sim_runs` mark's case count.

    Returns:
        The session's result, which must have passed every case.
    """
    if ini is not None:
        pytester.makeini(f"[pytest]\nsim_out = {ini}\n")
    pytester.makepyfile(
        test_scenario=_source(_case_test((f"sim_runs({runs})", "sim_duration_s(3)")))
    )
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=runs)
    return result


@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_session_with_no_sim_out_writes_to_the_default_run_folder(
    pytester: pytest.Pytester, xdist: tuple[str, ...]
):
    """Run files land in `<rootdir>/monte_carlo/<test file>/<test>/`, whatever the worker count.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    result = _default_session(pytester, *xdist)

    directory = pytester.path / "monte_carlo" / "test_scenario" / "test_case"
    assert [run["name"] for run in _read_runs(directory)] == ["run_0000.h5", "run_0001.h5"]
    assert len(list(pytester.path.rglob("run_*.h5"))) == 2
    result.stdout.fnmatch_lines([f"wise-dynamics: 2 run files in {directory}"])


@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_relative_sim_out_ini_setting_is_resolved_against_the_rootdir(
    pytester: pytest.Pytester, xdist: tuple[str, ...]
):
    """`sim_out = ini_out` puts the run folder at `<rootdir>/ini_out`, not `monte_carlo`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    _default_session(pytester, *xdist, ini="ini_out")

    directory = pytester.path / "ini_out" / "test_scenario" / "test_case"
    assert [run["name"] for run in _read_runs(directory)] == ["run_0000.h5", "run_0001.h5"]
    assert not (pytester.path / "monte_carlo").exists()


def test_an_absolute_sim_out_ini_setting_is_used_as_given(
    pytester: pytest.Pytester, tmp_path: Path
):
    """An absolute `sim_out` ini value names the run folder, outside the rootdir.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory, holding the run folder.
    """
    out = tmp_path / "absolute_out"
    _default_session(pytester, *XDIST, ini=str(out))

    assert [run["name"] for run in _read_runs(out / "test_scenario" / "test_case")] == [
        "run_0000.h5",
        "run_0001.h5",
    ]
    assert not list(pytester.path.rglob("run_*.h5"))


def test_a_sim_out_mark_overrides_the_ini_setting(pytester: pytest.Pytester):
    """A test with a `sim_out` mark writes there, and nothing goes to the ini setting's folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\nsim_out = ini_out\n")
    runs = _run_session(pytester, *SERIAL, runs=2, passed=2)

    assert [run["name"] for run in runs] == ["run_0000.h5", "run_0001.h5"]
    assert not (pytester.path / "ini_out").exists()
    assert not (pytester.path / "monte_carlo").exists()


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

    Each worker changes directory before the plugin configures it, and the run files still land
    under the rootdir's `out`.

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
    pytester.makepyfile(
        test_scenario=_source(_case_test(("sim_runs(2)", 'sim_out("out")', "sim_duration_s(3)")))
    )
    pytester.runpytest_subprocess("-n", "2").assert_outcomes(passed=2)

    in_out_dir = sorted((pytester.path / "out" / "test_scenario" / "test_case").glob("run_*.h5"))
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
    """Two tests differing in case count and output directory are two campaigns.

    Each test's run files are in its own directory, nothing is in either root, and the summary
    reports each directory.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    out_a, out_b = pytester.path / "out_a", pytester.path / "out_b"
    pytester.makepyfile(
        test_two=_source(
            _case_test(name="test_a", marks=("sim_runs(3)", _out_mark(out_a), "sim_duration_s(3)")),
            _case_test(name="test_b", marks=("sim_runs(2)", _out_mark(out_b), "sim_duration_s(3)")),
        )
    )
    result = pytester.runpytest_subprocess(*xdist)

    result.assert_outcomes(passed=5)
    dir_a, dir_b = out_a / "test_two" / "test_a", out_b / "test_two" / "test_b"
    assert [run["name"] for run in _read_runs(dir_a)] == [f"run_000{i}.h5" for i in range(3)]
    assert [run["name"] for run in _read_runs(dir_b)] == [f"run_000{i}.h5" for i in range(2)]
    assert len(list(pytester.path.rglob("run_*.h5"))) == 5
    result.stdout.fnmatch_lines(
        [f"wise-dynamics: 3 run files in {dir_a}", f"wise-dynamics: 2 run files in {dir_b}"]
    )


def test_test_files_outside_the_rootdir_are_named_by_their_stem(
    pytester: pytest.Pytester, tmp_path_factory: pytest.TempPathFactory
):
    """A test file outside `--rootdir` is named by its stem, so same-named tests do not collide.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path_factory: pytest's factory for directories outside the rootdir.
    """
    out = pytester.path / "out"
    source = _source(_case_test(("sim_runs(1)", _out_mark(out), "sim_duration_s(3)")))
    files = []
    for stem in ("test_one", "test_two"):
        files.append(tmp_path_factory.mktemp(stem) / f"{stem}.py")
        files[-1].write_text(source)
    result = pytester.runpytest_subprocess(
        "-p", "no:xdist", "--rootdir", str(pytester.path), *map(str, files)
    )

    result.assert_outcomes(passed=2)
    for stem in ("test_one", "test_two"):
        assert [run["name"] for run in _read_runs(out / stem / "test_case")] == ["run_0000.h5"]
    assert len(list(out.rglob("run_*.h5"))) == 2
    assert not [path for file in files for path in file.parent.rglob("run_*.h5")]


def test_a_class_and_a_nested_class_qualify_the_directory(pytester: pytest.Pytester):
    """A test's directory is its file path, each class outermost first, then its function.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    out_dir = pytester.path / "out"
    inner = textwrap.indent(_case_test(name="test_inner", method=True), "        ")
    outer = textwrap.indent(_case_test(name="test_outer", method=True), "    ")
    source = (
        SCENARIO_HEADER
        + f"\npytestmark = [pytest.mark.sim_runs(1), pytest.mark.{_out_mark(out_dir)},"
        " pytest.mark.sim_duration_s(3)]\n\n\n"
        + f"class TestOuter:{outer}\n    class TestInner:{inner}"
    )
    (pytester.path / "tests" / "sim").mkdir(parents=True)
    (pytester.path / "tests" / "sim" / "test_nested.py").write_text(source)
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    base = out_dir / "tests" / "sim" / "test_nested"
    assert sorted(path.relative_to(out_dir) for path in out_dir.rglob("run_*.h5")) == [
        base.relative_to(out_dir) / "TestOuter" / "TestInner" / "test_inner" / "run_0000.h5",
        base.relative_to(out_dir) / "TestOuter" / "test_outer" / "run_0000.h5",
    ]


def test_the_closest_mark_wins_over_class_and_module_marks(pytester: pytest.Pytester):
    """A mark on a function overrides the class's, which overrides the module's.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
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


@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_session_creates_nothing_in_the_temporary_directory(
    pytester: pytest.Pytester,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    xdist: tuple[str, ...],
):
    """A session that runs cases leaves an empty `TMPDIR` empty; its run files are in the rootdir.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory, holding the `TMPDIR` used.
        monkeypatch: pytest's fixture for setting `TMPDIR` for this test only.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    _default_session(pytester, f"--basetemp={tmp_path / 'basetemp'}", *xdist)

    assert list(temporary.iterdir()) == []
    assert len(list((pytester.path / "monte_carlo").rglob("run_*.h5"))) == 2


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
        ('sim_out("")', "sim_out"),
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


def test_the_five_marks_are_registered(pytester: pytest.Pytester):
    """`--strict-markers` accepts the five marks and rejects one the plugin does not register.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    result = _strict_session(pytester, ["sim_bogus(1)"])
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*'sim_bogus' not found in `markers` configuration option*"])

    marks = [
        'sim_scenario("leo500")',
        "sim_runs(1)",
        'sim_fidelity("lo")',
        "sim_duration_s(3)",
        _out_mark(pytester.path / "out"),
    ]
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
    pytester.makepyfile(
        f"import pytest\n\n\n{decorators}def test_case(sim_case_index):\n    pass\n"
    )
    return pytester.runpytest_subprocess("-p", "no:xdist", "--strict-markers")


def _populate_out(pytester: pytest.Pytester, runs: int, root: str = "mark") -> Path:
    """Write a `sim_runs(runs)` test whose run folder holds 5 stale run files.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        runs: The test's `sim_runs` mark.
        root: How the test names its run folder, a key of `ROOT_NAMES`: a `sim_out` mark, the
            default folder, or the `sim_out` ini setting.

    Returns:
        The test's directory, holding `run_0000.h5` to `run_0004.h5` (dummy) and `notes.txt`.
    """
    out_dir = pytester.path / ROOT_NAMES[root]
    test_dir = out_dir / "test_scenario" / "test_case"
    test_dir.mkdir(parents=True)
    for index in range(5):
        (test_dir / f"run_{index:04d}.h5").write_bytes(b"stale")
    (test_dir / "notes.txt").write_text("kept")
    if root == "ini":
        pytester.makeini(f"[pytest]\nsim_out = {ROOT_NAMES[root]}\n")
    marks = (f"sim_runs({runs})", "sim_duration_s(3)", *([_out_mark(out_dir)] * (root == "mark")))
    pytester.makepyfile(test_scenario=_source(_case_test(marks)))
    return test_dir


@pytest.mark.parametrize("root", ROOT_NAMES)
@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_run_folder_is_overwritten_each_run(
    pytester: pytest.Pytester, xdist: tuple[str, ...], root: str
):
    """Each case replaces its own run file and run files beyond the case count are removed.

    Other files in the directory stay, whether the folder is a mark's, the default or the ini
    setting's.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    test_dir = _populate_out(pytester, runs=2, root=root)
    pytester.runpytest_subprocess(*xdist).assert_outcomes(passed=2)

    runs = _read_runs(test_dir)
    assert [run["name"] for run in runs] == ["run_0000.h5", "run_0001.h5"]
    assert [run["attrs"]["case"] for run in runs] == [0, 1]
    assert (test_dir / "notes.txt").read_text() == "kept"


@pytest.mark.parametrize("root", ROOT_NAMES)
def test_rerunning_one_case_by_node_id_rewrites_only_its_run_file(
    pytester: pytest.Pytester, root: str
):
    """Re-running `test_case[1]` after a full run rewrites `run_0001.h5` and no other.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    test_dir = _populate_out(pytester, runs=3, root=root)
    pytester.runpytest_subprocess("-p", "no:xdist").assert_outcomes(passed=3)
    before = {path.name: path.stat().st_mtime_ns for path in test_dir.glob("run_*.h5")}
    assert sorted(before) == ["run_0000.h5", "run_0001.h5", "run_0002.h5"]

    result = pytester.runpytest_subprocess("-p", "no:xdist", "test_scenario.py::test_case[1]")
    result.assert_outcomes(passed=1)

    after = {path.name: path.stat().st_mtime_ns for path in test_dir.glob("run_*.h5")}
    assert sorted(after) == sorted(before)
    assert after["run_0001.h5"] > before["run_0001.h5"]
    assert after["run_0000.h5"] == before["run_0000.h5"]
    assert after["run_0002.h5"] == before["run_0002.h5"]


@pytest.mark.parametrize("root", ROOT_NAMES)
@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_collect_only_session_removes_nothing(
    pytester: pytest.Pytester, xdist: tuple[str, ...], root: str
):
    """`--collect-only` leaves every run file in the run folder, whoever named it.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    test_dir = _populate_out(pytester, runs=2, root=root)
    result = pytester.runpytest_subprocess(*xdist, "--collect-only", "-q")
    assert result.ret == pytest.ExitCode.OK
    assert sorted(path.name for path in test_dir.glob("run_*.h5")) == [
        f"run_{index:04d}.h5" for index in range(5)
    ]
    assert {path.read_bytes() for path in test_dir.glob("run_*.h5")} == {b"stale"}


@pytest.mark.parametrize("root", ROOT_NAMES)
def test_a_second_full_xdist_run_into_the_same_run_folder_completes(
    pytester: pytest.Pytester, root: str
):
    """Workers overwriting each other's previous run files do not race.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    test_dir = _populate_out(pytester, runs=4, root=root)
    for _ in range(2):
        pytester.runpytest_subprocess(*XDIST).assert_outcomes(passed=4)
    runs = _read_runs(test_dir)
    assert [run["attrs"]["case"] for run in runs] == [0, 1, 2, 3]


@pytest.mark.parametrize("root", ROOT_NAMES)
def test_ci_over_a_five_case_run_folder_keeps_two_fresh_files(pytester: pytest.Pytester, root: str):
    """`--ci` caps the cases at 2, so `run_0002.h5` to `run_0004.h5` are removed.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    test_dir = _populate_out(pytester, runs=5, root=root)
    pytester.runpytest_subprocess("-p", "no:xdist", "--ci").assert_outcomes(passed=2)
    runs = _read_runs(test_dir)
    assert [run["name"] for run in runs] == ["run_0000.h5", "run_0001.h5"]
    assert [run["attrs"]["case"] for run in runs] == [0, 1]


@pytest.mark.parametrize("root", ROOT_NAMES)
def test_every_run_folder_replaces_an_existing_run_file(pytester: pytest.Pytester, root: str):
    """Writing a case's run file twice replaces it, whoever named the run folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        root: How the test names its run folder, a key of `ROOT_NAMES`.
    """
    mark = f"@pytest.mark.{_out_mark(pytester.path / ROOT_NAMES[root])}\n" * (root == "mark")
    if root == "ini":
        pytester.makeini(f"[pytest]\nsim_out = {ROOT_NAMES[root]}\n")
    mark += "@pytest.mark.sim_runs(1)\n"
    pytester.makepyfile(
        SCENARIO_HEADER
        + f"""
{mark}def test_case(sim_case_index, sim_scenario, sim_run_file):
    datasets = sim_scenario.run(Guidance(), {{}}, duration_s=1)
    sim_run_file(datasets)
    sim_run_file(datasets)
"""
    )
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)


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


def _editable_direct_url(repository: Path) -> dict:
    """Build the `direct_url.json` of an editable install of `repository`.

    Args:
        repository: The checkout installed.

    Returns:
        The golden editable fixture, as uv writes it, with `repository` substituted.
    """
    text = (DATA / "direct_url_editable.json").read_text().replace("{repo}", str(repository))
    return json.loads(text)


def _provenance_session(
    pytester: pytest.Pytester, direct_url: dict | None, *args: str
) -> pytest.RunResult:
    """Run the two-case scenario with the plugin's `_direct_url` patched to `direct_url`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        direct_url: What `_direct_url` returns: the install's parsed `direct_url.json`, or `None`.
        *args: Further pytest arguments.

    Returns:
        The session's result.
    """
    pytester.makeconftest(
        "import wise_dynamics.pytest_plugin as plugin\n\n"
        f"plugin._direct_url = lambda: {direct_url!r}\n"
    )
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(2)", "sim_duration_s(3)"))))
    return pytester.runpytest_subprocess(*args)


def _provenance_attrs(pytester: pytest.Pytester) -> list[dict]:
    """Read each run file's four provenance attributes from the default run folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.

    Returns:
        One mapping per run file: `install`, `commit`, `requested_revision` and `dirty`.
    """
    prefix = "versions_wise_dynamics_"
    runs = _read_runs(pytester.path / "monte_carlo" / "test_scenario" / "test_case")
    return [
        {
            key: run["attrs"][prefix + key]
            for key in ("install", "commit", "requested_revision", "dirty")
        }
        for run in runs
    ]


def _assert_false(value: object) -> None:
    """Assert `value` is the boolean `False` as an HDF5 attribute reads back.

    Args:
        value: The attribute's value.
    """
    assert type(value) is np.bool_
    assert not value


def _assert_true(value: object) -> None:
    """Assert `value` is the boolean `True` as an HDF5 attribute reads back.

    Args:
        value: The attribute's value.
    """
    assert type(value) is np.bool_
    assert value


def test_the_golden_direct_urls_are_the_forms_uv_writes():
    """The two checked-in `direct_url.json` fixtures are the pinned and the editable forms."""
    vcs = json.loads((DATA / "direct_url_vcs.json").read_text())
    assert vcs["vcs_info"]["vcs"] == "git"
    assert re.fullmatch(r"[0-9a-f]{40}", vcs["vcs_info"]["commit_id"])
    editable = _editable_direct_url(Path("/some/checkout"))
    assert editable == {"url": "file:///some/checkout", "dir_info": {"editable": True}}


@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
def test_a_pinned_install_records_its_commit_as_pinned(
    pytester: pytest.Pytester, xdist: tuple[str, ...]
):
    """A VCS install records `pinned`, its commit, its requested revision and a clean tree.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    direct_url = json.loads((DATA / "direct_url_vcs.json").read_text())
    _provenance_session(pytester, direct_url, *xdist).assert_outcomes(passed=2)

    attrs = _provenance_attrs(pytester)
    assert len(attrs) == 2
    for found in attrs:
        assert found["install"] == "pinned"
        assert found["commit"] == direct_url["vcs_info"]["commit_id"]
        assert found["requested_revision"] == direct_url["vcs_info"]["requested_revision"]
        _assert_false(found["dirty"])


def test_a_pinned_install_with_no_requested_revision_records_an_empty_one(
    pytester: pytest.Pytester,
):
    """A VCS install that names no revision records `requested_revision` as an empty string.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    direct_url = json.loads((DATA / "direct_url_vcs.json").read_text())
    del direct_url["vcs_info"]["requested_revision"]
    _provenance_session(pytester, direct_url, *SERIAL).assert_outcomes(passed=2)

    attrs = _provenance_attrs(pytester)
    assert len(attrs) == 2
    for found in attrs:
        assert found["install"] == "pinned"
        assert found["commit"] == direct_url["vcs_info"]["commit_id"]
        assert found["requested_revision"] == ""


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
    direct_url = _editable_direct_url(repository)
    _provenance_session(pytester, direct_url, *XDIST).assert_outcomes(passed=2)

    attrs = _provenance_attrs(pytester)
    assert len(attrs) == 2
    for found in attrs:
        assert found["install"] == "local"
        assert found["commit"] == commit
        assert found["requested_revision"] == ""
        (_assert_true if dirty else _assert_false)(found["dirty"])


def test_wise_dynamics_own_editable_install_records_local(pytester: pytest.Pytester):
    """With nothing patched, this repository's editable install records `local` and its HEAD.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(1)", "sim_duration_s(3)"))))
    pytester.runpytest_subprocess(*SERIAL).assert_outcomes(passed=1)

    (found,) = _provenance_attrs(pytester)
    assert found["install"] == "local"
    assert found["commit"] == _git(REPOSITORY, "rev-parse", "HEAD")
    assert found["requested_revision"] == ""
    assert type(found["dirty"]) is np.bool_


def _plain_directory(tmp_path: Path) -> Path:
    """Create a directory that is not in any git repository.

    Args:
        tmp_path: The test's own temporary directory.

    Returns:
        The new directory.
    """
    directory = tmp_path / "plain"
    directory.mkdir()
    return directory


UNDETERMINABLE = {
    "no-direct-url": lambda tmp_path: None,
    "non-editable-directory": lambda tmp_path: {"url": tmp_path.as_uri(), "dir_info": {}},
    "non-git-vcs": lambda tmp_path: {
        "url": "https://example.org/repository",
        "vcs_info": {"vcs": "hg", "commit_id": "0" * 40},
    },
    "not-a-repository": lambda tmp_path: _editable_direct_url(_plain_directory(tmp_path)),
    "git-unavailable": lambda tmp_path: _editable_direct_url(REPOSITORY),
}
"""Each install the plugin cannot describe, as the `direct_url.json` a session is given."""


def _undeterminable_session(
    pytester: pytest.Pytester,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    *args: str,
) -> pytest.RunResult:
    """Run a session whose install `case` names, and check it stops with the install message.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory.
        monkeypatch: pytest's fixture for setting the session's environment for this test only.
        case: A key of `UNDETERMINABLE`.
        *args: Further pytest arguments.

    Returns:
        The session's result, whose exit code is a usage error, with no run file written.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    if case == "git-unavailable":
        (tmp_path / "no_git").mkdir()
        monkeypatch.setenv("PATH", str(tmp_path / "no_git"))
    result = _provenance_session(pytester, UNDETERMINABLE[case](tmp_path), *args)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    output = result.stdout.lines + result.stderr.lines
    assert len([line for line in output if "uv sync" in line]) == 1
    assert len([line for line in output if "uv pip install -e" in line]) == 1
    assert not list(pytester.path.rglob("run_*.h5"))
    return result


@pytest.mark.parametrize("xdist", [SERIAL, XDIST], ids=["serial", "xdist"])
@pytest.mark.parametrize("case", UNDETERMINABLE)
def test_an_install_that_cannot_be_described_stops_the_session(
    pytester: pytest.Pytester,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    xdist: tuple[str, ...],
):
    """An install the plugin cannot describe exits 4, names both supported installs, runs no case.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory.
        monkeypatch: pytest's fixture for setting the session's environment for this test only.
        case: A key of `UNDETERMINABLE`.
        xdist: The arguments that run the session with pytest-xdist or without it.
    """
    result = _undeterminable_session(pytester, tmp_path, monkeypatch, case, *xdist)
    result.assert_outcomes()


def test_an_install_that_cannot_be_described_stops_a_collect_only_session(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--collect-only` is a session too: a bad install stops it with exit 4.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory.
        monkeypatch: pytest's fixture for setting the session's environment for this test only.
    """
    result = _undeterminable_session(
        pytester, tmp_path, monkeypatch, "no-direct-url", *SERIAL, "--collect-only"
    )
    assert not [line for line in result.outlines if "test_scenario.py::" in line]


def test_the_plugin_runs_without_the_notebook_extra(pytester: pytest.Pytester):
    """With IPython and jupytext unimportable, a sim test still runs; the notebook module says why.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
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
    assert len(list((pytester.path / "monte_carlo").rglob("run_*.h5"))) == 1


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
    """The README's worked example, marks included, runs as a session and writes its run files.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    (example,) = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    pytester.makepyfile(test_example=example)
    pytester.runpytest_subprocess().assert_outcomes(passed=2)

    files = sorted((pytester.path / "runs" / "test_example" / "test_case").glob("run_*.h5"))
    assert len(files) == 2
    for path in files:
        with h5py.File(path, "r") as run_file:
            assert run_file["t_s"][:].tolist() == [0.0, 10.0, 20.0]
            assert run_file["guidance/sigma_RN"].attrs["units"] == "MRP"
            assert run_file.attrs["versions_guidance"] == "1.0"
