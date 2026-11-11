"""Sessions: `session_directory`, the per-case result and crash record, and `read_test_results`.

The plugin's behaviour is exercised through real pytest sessions in subprocesses. Cases write
minimal run files, so no Basilisk simulation runs.
"""

import importlib
import json
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

SESSION = "s"
NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"
SERIAL = ("-p", "no:xdist")
FULL = ("--full-monte-carlo",)
CASE_TEST = """
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.sim_runs(RUNS)
def test_case(sim_case_index, sim_run_file):
    mode = Path("mode").read_text() if Path("mode").exists() else "pass"
    if mode == "raise-before-write":
        raise RuntimeError("boom")
    if mode == "abort" and sim_case_index == 1:
        os.abort()
    sim_run_file({"t_s": (np.zeros(2), "s")})
    if mode == "assert-after-write":
        assert False, "late"
"""


def _sessions() -> ModuleType:
    """Import `wise_dynamics.sessions` when a test needs it, so its absence fails that test only.

    Returns:
        The module.
    """
    return importlib.import_module("wise_dynamics.sessions")


def _point(pytester: pytest.Pytester, name: str, root: str = "monte_carlo") -> Path:
    """Write the session pointer and return the session's folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: What the pointer holds, as written.
        root: The `sim_out` root, relative to the rootdir.

    Returns:
        `<rootdir>/<root>/<name>`.
    """
    (pytester.path / root).mkdir(parents=True, exist_ok=True)
    (pytester.path / root / "SESSION").write_text(name)
    return pytester.path / root / name.strip()


def _campaign(pytester: pytest.Pytester, runs: int) -> Path:
    """Write the one-test campaign into session `s` and return the test's directory.

    The test's behaviour follows the content of a file `mode` in the rootdir: absent or `pass`
    writes a run file; `raise-before-write` raises before it; `assert-after-write` fails after it;
    `abort` kills the process in case 1.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        runs: The test's `sim_runs`.

    Returns:
        The test's directory in the session.
    """
    session = _point(pytester, SESSION + "\n")
    pytester.makepyfile(test_scenario=CASE_TEST.replace("RUNS", str(runs)))
    return session / "test_scenario" / "test_case"


def _mode(pytester: pytest.Pytester, mode: str) -> None:
    """Set how the campaign's cases behave on the next run.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        mode: One of the modes `_campaign` lists.
    """
    (pytester.path / "mode").write_text(mode)


def test_session_directory_is_the_root_joined_to_the_pointers_name(pytester: pytest.Pytester):
    """The pointer's stripped content names the folder under the `sim_out` root.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester, "alpha-1.2\n")

    assert _sessions().session_directory(pytester.parseconfig()) == (
        pytester.path / "monte_carlo" / "alpha-1.2"
    )


def test_session_directory_follows_the_sim_out_ini_setting(pytester: pytest.Pytester):
    """A relative `sim_out` is resolved against the rootdir.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\nsim_out = ini_out\n")
    _point(pytester, "beta", root="ini_out")

    assert _sessions().session_directory(pytester.parseconfig()) == (
        pytester.path / "ini_out" / "beta"
    )


@pytest.mark.parametrize("content", [None, "", "  \n"], ids=["missing", "empty", "blank"])
def test_session_directory_with_no_active_session_raises_a_usage_error(
    pytester: pytest.Pytester, content: str | None
):
    """A missing, empty or blank pointer raises `UsageError` with the no-session message.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        content: What `SESSION` holds, or `None` for no file.
    """
    if content is not None:
        _point(pytester, content)
    config = pytester.parseconfig()

    with pytest.raises(pytest.UsageError) as raised:
        _sessions().session_directory(config)

    assert str(raised.value) == NO_SESSION


@pytest.mark.parametrize("name", ["..", "SESSION"])
def test_session_directory_with_an_invalid_name_raises_a_usage_error_naming_the_file(
    pytester: pytest.Pytester, name: str
):
    """An invalid name in the pointer raises `UsageError` naming `SESSION`, not the no-session one.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: The invalid name the pointer holds.
    """
    _point(pytester, name + "\n")
    config = pytester.parseconfig()

    with pytest.raises(pytest.UsageError) as raised:
        _sessions().session_directory(config)

    assert "SESSION" in str(raised.value)
    assert str(raised.value) != NO_SESSION


def test_a_case_that_raises_before_its_run_file_leaves_a_crash_record(pytester: pytest.Pytester):
    """A case failing before `sim_run_file` writes `run_NNNN.crash.json` and no run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 2)
    _mode(pytester, "raise-before-write")
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(failed=2)

    assert not list(directory.glob("run_*.h5"))
    for case in (0, 1):
        record = json.loads((directory / f"run_{case:04d}.crash.json").read_text())
        assert sorted(record) == ["exception", "traceback"]
        assert isinstance(record["exception"], str)
        assert isinstance(record["traceback"], str)
        assert "boom" in record["exception"]
        assert "boom" in record["traceback"]


def test_a_case_whose_setup_fails_leaves_a_crash_record(pytester: pytest.Pytester):
    """A failing fixture setup counts like a failing call: a record, no run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 1)
    pytester.makeconftest(
        """
        import pytest


        @pytest.fixture(autouse=True)
        def broken(sim_case_index):
            raise RuntimeError("setup boom")
        """
    )
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(errors=1)

    record = json.loads((directory / "run_0000.crash.json").read_text())
    assert "setup boom" in record["exception"]
    assert not list(directory.glob("run_*.h5"))


def test_a_passing_rerun_replaces_a_crash_with_a_run_file(pytester: pytest.Pytester):
    """After a crash, a case that passes leaves its run file and no crash record.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 1)
    _mode(pytester, "raise-before-write")
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(failed=1)
    assert (directory / "run_0000.crash.json").exists()

    _mode(pytester, "pass")
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=1)

    assert sorted(path.name for path in directory.iterdir()) == ["planned.json", "run_0000.h5"]


def test_a_crashing_rerun_replaces_a_run_file_with_a_crash_record(pytester: pytest.Pytester):
    """After a pass, a case that crashes leaves its crash record and no run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 1)
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=1)
    assert (directory / "run_0000.h5").exists()

    _mode(pytester, "raise-before-write")
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(failed=1)

    assert sorted(path.name for path in directory.iterdir()) == [
        "planned.json",
        "run_0000.crash.json",
    ]


def test_a_case_that_asserts_after_writing_keeps_its_run_file_only(pytester: pytest.Pytester):
    """A case that wrote its run file and then failed has the file and no crash record.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 1)
    _mode(pytester, "assert-after-write")
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(failed=1)

    assert sorted(path.name for path in directory.iterdir()) == ["planned.json", "run_0000.h5"]


def test_rerunning_one_case_replaces_only_its_own_result(pytester: pytest.Pytester):
    """A node-id rerun of case 1 rewrites its file and leaves case 0's file untouched.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 2)
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=2)
    before = {path.name: path.stat().st_mtime_ns for path in directory.glob("run_*.h5")}

    pytester.runpytest_subprocess(*FULL, *SERIAL, "test_scenario.py::test_case[1]").assert_outcomes(
        passed=1
    )

    after = {path.name: path.stat().st_mtime_ns for path in directory.glob("run_*.h5")}
    assert sorted(after) == ["run_0000.h5", "run_0001.h5"]
    assert after["run_0000.h5"] == before["run_0000.h5"]
    assert after["run_0001.h5"] != before["run_0001.h5"]


def test_a_worker_that_dies_in_a_case_leaves_a_crash_record(pytester: pytest.Pytester):
    """Under `-n 2`, `os.abort()` in case 1 gives it a crash record and no run file.

    Cases 0 and 2 run to a run file. The first run passes every case, so the crash also replaces
    case 1's earlier run file.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 3)
    pytester.runpytest_subprocess(*FULL, "-n", "2").assert_outcomes(passed=3)
    assert (directory / "run_0001.h5").exists()

    _mode(pytester, "abort")
    result = pytester.runpytest_subprocess(*FULL, "-n", "2")

    result.assert_outcomes(passed=2, failed=1)
    assert sorted(path.name for path in directory.iterdir()) == [
        "planned.json",
        "run_0000.h5",
        "run_0001.crash.json",
        "run_0002.h5",
    ]
    record = json.loads((directory / "run_0001.crash.json").read_text())
    assert sorted(record) == ["exception", "traceback"]
    assert isinstance(record["exception"], str)
    assert isinstance(record["traceback"], str)


def test_a_worker_that_dies_in_a_class_method_in_a_subdirectory_leaves_the_record_beside_its_runs(
    pytester: pytest.Pytester,
):
    """The crash record of a dead worker sits where the plugin writes that test's run files.

    The test is a method of a class in a file below the rootdir, so the controller's path from the
    node id must agree with the worker's path from the collected item.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester, SESSION + "\n")
    (pytester.path / "tests" / "sub").mkdir(parents=True)
    source = CASE_TEST.replace("RUNS", "3").replace("def test_case(", "def test_case(self, ")
    source = source.split("@pytest.mark.sim_runs", 1)
    body = "@pytest.mark.sim_runs" + source[1]
    (pytester.path / "tests" / "sub" / "test_x.py").write_text(
        source[0] + "class TestA:\n" + textwrap.indent(body, "    ")
    )
    _mode(pytester, "abort")

    pytester.runpytest_subprocess(*FULL, "-n", "2").assert_outcomes(passed=2, failed=1)

    directory = session / "tests" / "sub" / "test_x" / "TestA" / "test_case"
    assert (directory / "run_0001.crash.json").is_file()
    assert not (directory / "run_0001.h5").exists()


def test_a_worker_that_dies_in_a_fixture_replaces_the_cases_earlier_run_file(
    pytester: pytest.Pytester,
):
    """A rerun whose worker aborts before the test body leaves only the crash record for that case.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester, SESSION + "\n")
    pytester.makepyfile(
        test_scenario="""
        import os
        from pathlib import Path

        import numpy as np
        import pytest


        @pytest.fixture
        def doomed(sim_case_index):
            if Path("mode").exists() and sim_case_index == 1:
                os.abort()


        @pytest.mark.sim_runs(3)
        def test_case(sim_case_index, sim_run_file, doomed):
            sim_run_file({"t_s": (np.zeros(2), "s")})
        """
    )
    directory = session / "test_scenario" / "test_case"
    pytester.runpytest_subprocess(*FULL, "-n", "2").assert_outcomes(passed=3)
    assert (directory / "run_0001.h5").exists()

    _mode(pytester, "abort")
    pytester.runpytest_subprocess(*FULL, "-n", "2").assert_outcomes(passed=2, failed=1)

    assert (directory / "run_0001.crash.json").is_file()
    assert not (directory / "run_0001.h5").exists()


def test_recorded_tests_lists_each_tests_folder_with_its_recorded_node_path(
    pytester: pytest.Pytester,
):
    """A session the plugin wrote lists exactly its tests' folders with the node paths recorded.

    One test is a plain function in a subdirectory, one a method of a class; a folder without
    results beside them is not listed.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    session = _point(pytester, SESSION + "\n")
    (pytester.path / "tests").mkdir()
    (pytester.path / "tests" / "test_b.py").write_text(CASE_TEST.replace("RUNS", "1"))
    method = CASE_TEST.replace("RUNS", "1").replace("def test_case(", "def test_case(self, ")
    head, body = method.split("@pytest.mark.sim_runs", 1)
    (pytester.path / "test_a.py").write_text(
        head + "class TestA:\n" + textwrap.indent("@pytest.mark.sim_runs" + body, "    ")
    )
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=2)
    (session / "test_a" / "stray").mkdir()

    folders = _sessions().recorded_tests(session)

    assert [(folder.directory, folder.nodeid) for folder in folders] == [
        (session / "test_a" / "TestA" / "test_case", "test_a.py::TestA::test_case"),
        (session / "tests" / "test_b" / "test_case", "tests/test_b.py::test_case"),
    ]


PLAIN_ABORT = """
import os

import pytest


@pytest.mark.parametrize("x", [3])
def test_plain(x):
    os.abort()
"""


@pytest.mark.parametrize("pointer", [False, True], ids=["no-pointer", "pointer"])
def test_a_worker_that_dies_in_a_test_that_is_not_a_case_leaves_no_crash_record(
    pytester: pytest.Pytester, pointer: bool
):
    """A crash in a test with no Monte Carlo case writes no record and creates no folder.

    The test's parameter id is a number, as a case's is. The run's failure is the crash itself,
    never a "no session" error.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        pointer: Whether a session is active.
    """
    session = _point(pytester, SESSION + "\n") if pointer else None
    pytester.makepyfile(test_plain=PLAIN_ABORT)

    result = pytester.runpytest_subprocess(*FULL, "-n", "1")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "crashed" in result.stdout.str()
    assert NO_SESSION not in result.stdout.str() + result.stderr.str()
    assert not list(pytester.path.rglob("*.crash.json"))
    if session is None:
        assert not (pytester.path / "monte_carlo").exists()
    else:
        assert not session.exists()


def test_a_strict_xfail_case_that_passes_without_a_run_file_is_no_internal_error(
    pytester: pytest.Pytester,
):
    """A Monte Carlo case that passes under a strict xfail fails normally; the run does not crash.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester, SESSION + "\n")
    pytester.makepyfile(
        test_xfail="""
        import pytest


        @pytest.mark.xfail(strict=True)
        @pytest.mark.sim_runs(1)
        def test_case(sim_case_index):
            pass
        """
    )

    result = pytester.runpytest_subprocess(*FULL, *SERIAL)

    output = result.stdout.str() + result.stderr.str()
    assert "INTERNALERROR" not in output
    result.assert_outcomes(failed=1)


def test_read_test_results_returns_cases_below_planned_only(tmp_path: Path):
    """Runs and crashes for cases `0..planned-1` are returned; a stale higher file is ignored.

    Args:
        tmp_path: The test's own temporary directory, holding a hand-built test directory.
    """
    directory = tmp_path / "test_case"
    directory.mkdir()
    (directory / "planned.json").write_text(json.dumps({"runs": 4}))
    for case in (0, 1, 3, 9):
        (directory / f"run_{case:04d}.h5").write_bytes(b"")
    crash = {"exception": "boom", "traceback": "Traceback..."}
    (directory / "run_0002.crash.json").write_text(json.dumps(crash))
    (directory / "run_0007.crash.json").write_text(json.dumps(crash))

    results = _sessions().read_test_results(directory)

    assert results.planned == 4
    assert results.runs == {case: directory / f"run_{case:04d}.h5" for case in (0, 1, 3)}
    assert results.crashes == {2: crash}


def test_read_test_results_of_a_session_written_by_the_plugin(pytester: pytest.Pytester):
    """The names the plugin writes are the names `read_test_results` reads.

    A full pass reads back as three runs, no crash, and a stale case 3 is ignored.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 3)
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=3)
    (directory / "run_0003.h5").write_bytes(b"")

    results = _sessions().read_test_results(directory)

    assert (results.planned, sorted(results.runs), results.crashes) == (3, [0, 1, 2], {})


@pytest.mark.parametrize("workers", [SERIAL, ("-n", "2")], ids=["serial", "xdist"])
def test_a_campaign_deletes_its_own_run_and_crash_files_numbered_at_or_above_its_plan(
    pytester: pytest.Pytester, workers: tuple[str, ...]
):
    """A folder with cases 0-9, then a run planned for 4: cases 4-9 go, 0-3 stay, siblings stay.

    Only case 0 is selected, so cases 1-3 are neither rerun nor rewritten. Another test's folder,
    which this run does not collect, is not touched.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        workers: The pytest-xdist arguments of the run.
    """
    directory = _campaign(pytester, 4)
    sibling = directory.parent / "test_gone"
    directory.mkdir(parents=True)
    sibling.mkdir()
    sessions = _sessions()
    for case in range(10):
        sessions.run_path(directory, case).write_bytes(b"old")
    for case in (1, 5, 8):
        sessions.crash_path(directory, case).write_text("{}")
    for path in (
        sessions.run_path(sibling, 7),
        sessions.crash_path(sibling, 7),
        sessions.run_path(sibling, 2),
    ):
        path.write_bytes(b"old")

    pytester.runpytest_subprocess(
        *FULL, *workers, "test_scenario.py::test_case[0]"
    ).assert_outcomes(passed=1)

    assert sorted(path.name for path in directory.iterdir()) == [
        "planned.json",
        "run_0000.h5",
        "run_0001.crash.json",
        "run_0001.h5",
        "run_0002.h5",
        "run_0003.h5",
    ]
    assert (directory / "run_0001.h5").read_bytes() == b"old"
    assert json.loads((directory / "planned.json").read_text())["runs"] == 4
    assert sorted(path.name for path in sibling.iterdir()) == [
        "run_0002.h5",
        "run_0007.crash.json",
        "run_0007.h5",
    ]


def test_a_collect_only_run_deletes_no_leftover_files(pytester: pytest.Pytester):
    """`--collect-only` clears nothing: only a run that writes `planned.json` clears leftovers.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    directory = _campaign(pytester, 2)
    directory.mkdir(parents=True)
    (directory / "run_0005.h5").write_bytes(b"old")

    pytester.runpytest_subprocess(*FULL, *SERIAL, "--collect-only", "-q")

    assert (directory / "run_0005.h5").exists()


DIRECTORY_TEST = """
from pathlib import Path

import numpy as np


def test_case(sim_case_index, sim_run_file, sim_test_directory):
    assert isinstance(sim_test_directory, Path)
    written = sim_run_file({"t_s": (np.zeros(2), "s")})
    assert written.parent == sim_test_directory
    Path("reported").write_text(str(sim_test_directory))
"""


@pytest.mark.parametrize(
    ("flags", "session"),
    [(("--full-monte-carlo",), SESSION), ((), "scratch")],
    ids=["full", "capped"],
)
def test_sim_test_directory_is_where_the_tests_run_files_are_written(
    pytester: pytest.Pytester, flags: tuple[str, ...], session: str
):
    """The fixture is the test's directory in the session the runs land in, `scratch` if capped.

    The test also checks, inside the run, that `sim_run_file` writes directly in it.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        flags: The arguments that make the run full or capped.
        session: The session the runs must land in.
    """
    _point(pytester, SESSION + "\n")
    pytester.makepyfile(
        test_scenario="import pytest\n"
        + DIRECTORY_TEST.replace("def test_case", "@pytest.mark.sim_runs(1)\ndef test_case")
    )

    pytester.runpytest_subprocess(*SERIAL, *flags).assert_outcomes(passed=1)

    expected = pytester.path / "monte_carlo" / session / "test_scenario" / "test_case"
    assert (pytester.path / "reported").read_text() == str(expected)
    assert (expected / "run_0000.h5").is_file()


def test_session_directory_is_scratch_whenever_wise_dynamics_capped_is_set(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    """With the variable non-empty, the answer is `<root>/scratch` whatever the pointer says.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture for setting the variable for this test only.
    """
    monkeypatch.setenv("WISE_DYNAMICS_CAPPED", "1")
    config = pytester.parseconfig()
    scratch = pytester.path / "monte_carlo" / "scratch"

    assert _sessions().session_directory(config) == scratch

    _point(pytester, "alpha\n")
    assert _sessions().session_directory(pytester.parseconfig()) == scratch


def test_an_empty_wise_dynamics_capped_leaves_the_pointers_session(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    """Only a non-empty value caps: an empty one is as good as unset.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture for setting the variable for this test only.
    """
    monkeypatch.setenv("WISE_DYNAMICS_CAPPED", "")
    _point(pytester, "alpha\n")

    assert _sessions().session_directory(pytester.parseconfig()) == (
        pytester.path / "monte_carlo" / "alpha"
    )


@pytest.fixture
def flown_session(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fly one full case into session `s` of a `sim_out` named `out`, then work in a nested cwd.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture; it unsets the variable and moves the working directory.

    Returns:
        The session's directory, which holds the run file.
    """
    monkeypatch.delenv("WISE_DYNAMICS_CAPPED", raising=False)
    pytester.makeini("[pytest]\nsim_out = out\n")
    session = _point(pytester, SESSION + "\n", root="out")
    pytester.makepyfile(test_scenario=CASE_TEST.replace("RUNS", "1"))
    pytester.runpytest_subprocess(*FULL, *SERIAL).assert_outcomes(passed=1)
    assert (session / "test_scenario" / "test_case" / "run_0000.h5").is_file()
    nested = pytester.path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    return session


def test_current_session_is_where_a_full_run_wrote_from_a_nested_directory(flown_session: Path):
    """Outside pytest, `current_session()` finds the rootdir and `sim_out` as pytest does.

    Args:
        flown_session: The session a full run wrote into; the working directory is below the
            rootdir.
    """
    assert _sessions().current_session() == flown_session


def test_current_session_is_scratch_when_wise_dynamics_capped_is_set(
    flown_session: Path, monkeypatch: pytest.MonkeyPatch
):
    """The variable reaches the subprocess that finds the session, so it answers `scratch`.

    Args:
        flown_session: The session a full run wrote into, beside which `scratch` is looked for.
        monkeypatch: pytest's fixture for setting the variable for this test only.
    """
    monkeypatch.setenv("WISE_DYNAMICS_CAPPED", "1")

    assert _sessions().current_session() == flown_session.parent / "scratch"


def test_current_session_with_no_pointer_raises_no_session_error(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    """With no pointer there is no session to return.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture; it unsets the variable.
    """
    monkeypatch.delenv("WISE_DYNAMICS_CAPPED", raising=False)
    pytester.makeini("[pytest]\nsim_out = out\n")

    with pytest.raises(_sessions().NoSessionError):
        _sessions().current_session()


def test_current_session_with_an_invalid_pointer_name_raises_runtime_error_with_pytests_output(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    """A pointer pytest rejects fails the subprocess, so no path is returned and the cause shows.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        monkeypatch: pytest's fixture; it unsets the variable.
    """
    monkeypatch.delenv("WISE_DYNAMICS_CAPPED", raising=False)
    _point(pytester, "bad/name\n")

    with pytest.raises(RuntimeError, match="could not report the session") as raised:
        _sessions().current_session()

    assert not isinstance(raised.value, _sessions().NoSessionError)
    assert "bad/name" in str(raised.value)
