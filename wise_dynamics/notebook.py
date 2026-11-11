"""Monte Carlo campaigns from a jupytext-paired notebook: the `%%monte_carlo` magic and `load_runs`.

`%load_ext wise_dynamics.notebook` registers the magic. It needs the `notebook` extra of
`wise-dynamics`. `wise_dynamics/basilisk/README.md` documents both.
"""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import pytest

try:
    import jupytext
    import jupytext.cli
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import Pretty, display
    from tqdm import tqdm
except ImportError as error:
    raise ImportError(
        "wise_dynamics.notebook needs IPython, jupytext and tqdm; "
        "install them with the extra: pip install 'wise-dynamics[notebook]'"
    ) from error

from wise_dynamics import sessions
from wise_dynamics.pytest_plugin import _active_session, _run_dir
from wise_dynamics.sessions import NoSessionError

__all__ = ["load_ipython_extension", "load_runs"]

_REPORT = "WISE_DYNAMICS_RUN_DIR_REPORT"


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register the `%%monte_carlo` cell magic.

    Args:
        ipython: The interactive shell loading the extension.
    """

    def monte_carlo(line: str, cell: str) -> None:
        """Sync the notebook's pair, run the cell, then run its top-level tests with `-n auto`.

        Before running, an `.ipynb`'s code cells after its last `%%monte_carlo` cell are marked
        notebook-only and the pair re-synced, so the workers importing the `.py` skip them.

        Args:
            line: The rest of the magic line, optionally the notebook's path.
            cell: The cell's source after the magic line.

        Raises:
            RuntimeError: The notebook is unpaired or will not sync, the cell differs from the
                notebook on disk or defines or creates no test, or the cell or a test fails.
        """
        notebook = _notebook_path(line)
        py = _notebook_py(notebook)
        if py != notebook:
            _sync(notebook)
        saved = {
            "\n".join(map(str.rstrip, each.source.partition("\n")[2].splitlines())).strip("\n")
            for each in jupytext.read(notebook).cells
            if _is_campaign(each)
        }
        if "\n".join(map(str.rstrip, cell.splitlines())).strip("\n") not in saved:
            raise RuntimeError(
                "This cell has unsaved changes. Save the notebook and run the cell again."
            )
        tests = [
            node.name
            for node in ast.parse(ipython.input_transformer_manager.transform_cell(cell)).body
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name.startswith("test")
            )
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        ]
        if notebook.suffix == ".ipynb":
            _mark_notebook_only(notebook)
        before = dict(ipython.user_ns)
        ipython.run_cell(cell).raise_error()
        tests += [
            name
            for name, value in ipython.user_ns.items()
            if name.startswith("test")
            and callable(value)
            and before.get(name) is not value
            and name not in tests
        ]
        if not tests:
            raise RuntimeError("This cell defines no test.")
        ci = ["--ci"] if os.environ.get("CI") else []
        progress = _Progress()
        exit_code = pytest.main(
            [
                *(f"{py}::{name}" for name in tests),
                "-n",
                "auto",
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
                *ci,
            ],
            plugins=[progress],
        )
        progress.update(force=True)
        if exit_code != pytest.ExitCode.OK:
            raise RuntimeError(f"pytest exited with {exit_code!r}")

    ipython.register_magic_function(monte_carlo, "cell", "monte_carlo")


def load_runs(
    test: str | Callable[..., object], notebook: str | os.PathLike[str] | None = None
) -> list[dict[str, Any]]:
    """Read back the run files of a notebook test in the active session, finished or not.

    It prints one line, `<session>: <found> of <planned> cases`, followed by `, crashed: <cases>`
    when any case left a crash record.

    Args:
        test: The test function or method defined in the notebook, or its node path in the paired
            `.py`, such as `"test_x"` or `"TestA::test_x"`.
        notebook: The `.ipynb` or `.py`; by default the notebook the magic would use.

    Returns:
        One record per run file, sorted by `attrs["case"]`: `attrs` holds the root attributes and
        `datasets` maps each HDF5 dataset path to `(values, units)`.

    Raises:
        RuntimeError: No Monte Carlo session is active, the test does not collect from the
            notebook's `.py`, or the node collects tests of more than one function.
        FileNotFoundError: The test has no results in the active session.
    """
    node = test if isinstance(test, str) else test.__qualname__.replace(".", "::")
    py = _notebook_py(_notebook_path(os.fspath(notebook) if notebook is not None else ""))
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "run_dir.json"
        collected = subprocess.run(
            [sys.executable, "-m", "pytest", f"{py}::{node}", "--collect-only", "-q"]
            + ["-o", "addopts=", "-p", "no:xdist", "-p", "wise_dynamics.notebook"]
            + (["--ci"] if os.environ.get("CI") else []),
            cwd=py.parent,
            env={**os.environ, _REPORT: str(report)},
            capture_output=True,
            text=True,
        )
        if collected.returncode or not report.exists():
            raise RuntimeError(
                f"{py}::{node} does not collect:\n{collected.stdout}{collected.stderr}"
            )
        handshake = json.loads(report.read_text())
    if handshake["session"] is None:
        raise RuntimeError(sessions.NO_SESSION)
    directories = handshake["directories"]
    if len(directories) != 1:
        raise RuntimeError(
            f"{py}::{node} collects tests in {len(directories)} run directories; it must name a "
            "single test function or method"
        )
    directory = Path(directories[0])
    if not (directory / sessions.PLANNED).is_file():
        raise FileNotFoundError(f"no Monte Carlo results in {directory}")
    results = sessions.read_test_results(directory)
    crashed = (
        f", crashed: {', '.join(map(str, sorted(results.crashes)))}" if results.crashes else ""
    )
    print(f"{handshake['session']}: {len(results.runs)} of {results.planned} cases{crashed}")
    runs = []
    for case in sorted(results.runs):
        file = results.runs[case]
        with h5py.File(file, "r") as run_file:
            names: list[str] = []
            run_file.visit(names.append)
            datasets = {
                name: (run_file[name][...], run_file[name].attrs["units"])
                for name in names
                if isinstance(run_file[name], h5py.Dataset)
            }
            runs.append({"attrs": dict(run_file.attrs), "datasets": datasets})
    return runs


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write the active session and the collected tests' run directories to the report file.

    The file is the one `WISE_DYNAMICS_RUN_DIR_REPORT` names; its JSON is `{"session": <name or
    null>, "directories": [...]}`, with no directories when there is no active session.

    An invalid name in the session pointer stops the collection with a usage error.

    Args:
        session: The finished collection session.
    """
    report = os.environ.get(_REPORT)
    if report and session.items:
        try:
            name: str | None = _active_session(session.config).name
        except NoSessionError:
            name = None
        directories = (
            [] if name is None else sorted({str(_run_dir(item)) for item in session.items})
        )
        Path(report).write_text(json.dumps({"session": name, "directories": directories}))


class _Progress:
    """A pytest plugin that shows the count of finished tests as one display, updated in place.

    The total is the number of collected tests; the text follows tqdm's meter, with the number of
    failed reports in any phase as its postfix. The display is redrawn about once a second.
    """

    def __init__(self) -> None:
        """Start the clock and show an empty meter."""
        self.total: int | None = None
        self.done = 0
        self.failed = 0
        self.start = self.shown = time.monotonic()
        self.handle = display(Pretty(self._text()), display_id=True)

    def _text(self) -> str:
        """Format the meter.

        Returns:
            tqdm's meter for the tests finished so far.
        """
        return tqdm.format_meter(
            self.done,
            self.total,
            time.monotonic() - self.start,
            unit="test",
            postfix=f"failed={self.failed}",
        )

    def update(self, force: bool = False) -> None:
        """Redraw the display, unless it was redrawn within the last second.

        Args:
            force: Redraw whatever the time since the last redraw.
        """
        if force or time.monotonic() - self.shown >= 1:
            self.shown = time.monotonic()
            self.handle.update(Pretty(self._text()))

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Set the total from the collected items when pytest-xdist is off.

        Args:
            session: The finished collection session.
        """
        if session.items:
            self.total = len(session.items)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: list[str]) -> None:
        """Set the total from the first worker's collected ids.

        Args:
            node: The pytest-xdist worker node that finished collecting.
            ids: The node ids it collected.
        """
        if self.total is None:
            self.total = len(ids)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count a failed report, and a finished test at its teardown.

        Args:
            report: One phase's report of a test.
        """
        if report.failed:
            self.failed += 1
        if report.when == "teardown":
            self.done += 1
        self.update()


def _notebook_path(line: str) -> Path:
    """Find the notebook: the argument, else `__vsc_ipynb_file__`, else `JPY_SESSION_NAME`.

    Args:
        line: The argument, possibly empty.

    Returns:
        The notebook's path, an `.ipynb` or the `.py` jupytext opened as a notebook.

    Raises:
        RuntimeError: None of the three names a notebook.
    """
    user_ns = getattr(get_ipython(), "user_ns", {})
    found = line.strip() or user_ns.get("__vsc_ipynb_file__") or os.environ.get("JPY_SESSION_NAME")
    if not found:
        raise RuntimeError(
            "The notebook's path is unknown; pass it as an argument, such as "
            "`%%monte_carlo path/to/notebook.ipynb`."
        )
    return Path(found)


def _sync(notebook: Path) -> None:
    """Sync a notebook with its pair through jupytext's CLI, the newer file's inputs winning.

    Args:
        notebook: The paired notebook.

    Raises:
        RuntimeError: jupytext could not sync it.
    """
    if jupytext.cli.jupytext(["--sync", "--quiet", str(notebook)]):
        raise RuntimeError(f"jupytext could not sync {notebook}")


def _is_campaign(cell: Any) -> bool:
    """Tell whether a notebook cell is a `%%monte_carlo` cell.

    Args:
        cell: A cell of a notebook jupytext read.

    Returns:
        Whether it is a code cell whose first line is the magic.
    """
    return cell.cell_type == "code" and cell.source.startswith("%%monte_carlo")


def _mark_notebook_only(notebook: Path) -> None:
    """Mark each code cell after the last `%%monte_carlo` cell `active="ipynb"`, then re-sync.

    A cell that already has `active` metadata keeps it, and a notebook with no cell to mark is not
    rewritten.

    Args:
        notebook: The saved, synced `.ipynb`, holding at least one `%%monte_carlo` cell.
    """
    saved = jupytext.read(notebook)
    last = max(i for i, each in enumerate(saved.cells) if _is_campaign(each))
    unmarked = [
        each
        for each in saved.cells[last + 1 :]
        if each.cell_type == "code" and "active" not in each.metadata
    ]
    if not unmarked:
        return
    for each in unmarked:
        each.metadata["active"] = "ipynb"
    jupytext.write(saved, notebook)
    _sync(notebook)


def _notebook_py(notebook: Path) -> Path:
    """Find the `.py` whose tests a notebook runs.

    Args:
        notebook: The `.ipynb`, or the `.py` jupytext opened as a notebook.

    Returns:
        `notebook` itself when it is a `.py`, else the `.py` among the paths jupytext pairs with
        it, from the notebook's metadata or a `jupytext.toml`.

    Raises:
        RuntimeError: An `.ipynb` has no paired `.py`.
    """
    if notebook.suffix == ".py":
        return notebook
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        jupytext.cli.jupytext(["--paired-paths", str(notebook)])
    pys = [line for line in out.getvalue().splitlines() if line.endswith(".py")]
    if not pys:
        raise RuntimeError(
            f"{notebook} has no paired .py; pair it with "
            "`jupytext --set-formats ipynb,py:percent <notebook>`"
        )
    return Path(pys[0])
