"""The pytest plugin, exercised through real pytest sessions in subprocesses.

Each session loads the plugin by its `pytest11` entry point. Each session's test file carries its
own marks in source.
"""

import re
import textwrap
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest

from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

README = Path(__file__).parent.parent / "wise_dynamics" / "basilisk" / "README.md"

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


def test_workers_share_the_default_out_dir(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    """pytest-xdist workers write to the one default root the controller chose.

    `TMPDIR` points the default root into this test's own directory, where it can be found.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture for setting `TMPDIR` for this test only.
    """
    monkeypatch.setenv("TMPDIR", str(pytester.path))
    pytester.makepyfile(test_scenario=_source(_case_test(("sim_runs(2)", "sim_duration_s(3)"))))
    result = pytester.runpytest_subprocess("-n", "2")

    result.assert_outcomes(passed=2)
    files = list(pytester.path.rglob("run_*.h5"))
    assert len(files) == 2
    directory = files[0].parent
    assert {path.parent for path in files} == {directory}
    assert directory.relative_to(pytester.path).parts[1:] == ("test_scenario", "test_case")
    assert directory.parent.parent.name.startswith("wise_dynamics_")
    result.stdout.fnmatch_lines([f"wise-dynamics: 2 run files in {directory}"])


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


def test_a_session_with_no_case_creates_nothing_in_the_temporary_directory(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A session with no case leaves `TMPDIR` as it found it.

    The default output directory sits under `TMPDIR`.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        tmp_path: The test's own temporary directory, holding the `TMPDIR` used.
        monkeypatch: pytest's fixture for setting `TMPDIR` for this test only.
    """
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    pytester.makepyfile("def test_nothing():\n    pass\n")
    before = sorted(temporary.iterdir())
    pytester.runpytest_subprocess(f"--basetemp={tmp_path / 'basetemp'}").assert_outcomes(passed=1)
    assert sorted(temporary.iterdir()) == before


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


def test_a_test_directory_holding_run_files_fails_that_tests_collection(
    pytester: pytest.Pytester,
):
    """A test directory already holding run files fails the test's collection, naming it.

    Run files in the output root itself, which is no test's directory, do not.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    out_dir = pytester.path / "out"
    stale_dir = out_dir / "test_scenario" / "test_case"
    stale_dir.mkdir(parents=True)
    (stale_dir / "run_0000.h5").write_bytes(b"")
    pytester.makepyfile(
        test_scenario=_source(_case_test(("sim_runs(1)", _out_mark(out_dir), "sim_duration_s(3)")))
    )
    result = pytester.runpytest_subprocess("-p", "no:xdist")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([f"*test_scenario.py::test_case: {stale_dir} already holds*"])

    (stale_dir / "run_0000.h5").unlink()
    (out_dir / "run_0000.h5").write_bytes(b"")
    pytester.runpytest_subprocess("-p", "no:xdist").assert_outcomes(passed=1)


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
