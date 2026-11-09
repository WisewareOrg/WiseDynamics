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
    from tqdm.auto import tqdm
except ImportError as error:
    raise ImportError(
        "wise_dynamics.notebook needs IPython, jupytext and tqdm; "
        "install them with the extra: pip install 'wise-dynamics[notebook]'"
    ) from error

from wise_dynamics.pytest_plugin import _run_dir

__all__ = ["load_ipython_extension", "load_runs"]

_REPORT = "WISE_DYNAMICS_RUN_DIR_REPORT"


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register the `%%monte_carlo` cell magic.

    Args:
        ipython: The interactive shell loading the extension.
    """

    def monte_carlo(line: str, cell: str) -> None:
        """Sync the notebook's pair, run the cell, then run its top-level tests with `-n auto`.

        Args:
            line: The rest of the magic line, optionally the notebook's path.
            cell: The cell's source after the magic line.

        Raises:
            RuntimeError: The notebook is unpaired or will not sync, the cell differs from the
                notebook on disk or defines no test, or the cell or a test fails.
        """
        notebook = _notebook_path(line)
        py = _notebook_py(notebook)
        if py != notebook and jupytext.cli.jupytext(["--sync", "--quiet", str(notebook)]):
            raise RuntimeError(f"jupytext could not sync {notebook}")
        saved = {
            "\n".join(map(str.rstrip, each.source.partition("\n")[2].splitlines())).strip("\n")
            for each in jupytext.read(notebook).cells
            if each.cell_type == "code" and each.source.startswith("%%monte_carlo")
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
        if not tests:
            raise RuntimeError("This cell defines no test.")
        ipython.run_cell(cell).raise_error()
        ci = ["--ci"] if os.environ.get("CI") else []
        bar = tqdm(unit="test")
        exit_code = pytest.main(
            [
                *(f"{py}::{name}" for name in tests),
                "-n",
                "auto",
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
                *ci,
            ],
            plugins=[_Progress(bar)],
        )
        bar.close()
        if exit_code != pytest.ExitCode.OK:
            raise RuntimeError(f"pytest exited with {exit_code!r}")

    ipython.register_magic_function(monte_carlo, "cell", "monte_carlo")


def load_runs(
    test: str | Callable[..., object], notebook: str | os.PathLike[str] | None = None
) -> list[dict[str, Any]]:
    """Read back the run files a notebook test wrote.

    Args:
        test: The test function or method defined in the notebook, or its node path in the paired
            `.py`, such as `"test_x"` or `"TestA::test_x"`.
        notebook: The `.ipynb` or `.py`; by default the notebook the magic would use.

    Returns:
        One record per run file, sorted by `attrs["case"]`: `attrs` holds the root attributes and
        `datasets` maps each HDF5 dataset path to `(values, units)`.

    Raises:
        RuntimeError: The test does not collect from the notebook's `.py`, or the node collects
            tests of more than one function.
        FileNotFoundError: The test's run directory holds no run file.
    """
    node = test if isinstance(test, str) else test.__qualname__.replace(".", "::")
    py = _notebook_py(_notebook_path(os.fspath(notebook) if notebook is not None else ""))
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "run_dir.json"
        collected = subprocess.run(
            [sys.executable, "-m", "pytest", f"{py}::{node}", "--collect-only", "-q"]
            + ["-o", "addopts=", "-p", "no:xdist", "-p", "wise_dynamics.notebook"],
            cwd=py.parent,
            env={**os.environ, _REPORT: str(report)},
            capture_output=True,
            text=True,
        )
        if collected.returncode or not report.exists():
            raise RuntimeError(
                f"{py}::{node} does not collect:\n{collected.stdout}{collected.stderr}"
            )
        directories = json.loads(report.read_text())
    if len(directories) != 1:
        raise RuntimeError(
            f"{py}::{node} collects tests in {len(directories)} run directories; it must name a "
            "single test function or method"
        )
    files = sorted(Path(directories[0]).glob("run_*.h5"))
    if not files:
        raise FileNotFoundError(f"no run files in {directories[0]}")
    runs = []
    for file in files:
        with h5py.File(file, "r") as run_file:
            names: list[str] = []
            run_file.visit(names.append)
            datasets = {
                name: (run_file[name][...], run_file[name].attrs["units"])
                for name in names
                if isinstance(run_file[name], h5py.Dataset)
            }
            runs.append({"attrs": dict(run_file.attrs), "datasets": datasets})
    return sorted(runs, key=lambda run: run["attrs"]["case"])


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write the collected tests' run directories to the file `WISE_DYNAMICS_RUN_DIR_REPORT` names.

    Args:
        session: The finished collection session.
    """
    report = os.environ.get(_REPORT)
    if report and session.items:
        Path(report).write_text(json.dumps(sorted({str(_run_dir(item)) for item in session.items})))


class _Progress:
    """A pytest plugin that counts finished tests on a progress bar.

    The bar's total is the number of collected tests, and its postfix the number of failed
    reports in any phase.
    """

    def __init__(self, bar: tqdm) -> None:
        """Hold the bar.

        Args:
            bar: A `tqdm` bar with no total yet.
        """
        self.bar = bar
        self.failed = 0

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Set the total from the collected items when pytest-xdist is off.

        Args:
            session: The finished collection session.
        """
        if session.items:
            self.bar.reset(total=len(session.items))

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: list[str]) -> None:
        """Set the total from the first worker's collected ids.

        Args:
            node: The pytest-xdist worker node that finished collecting.
            ids: The node ids it collected.
        """
        if self.bar.total is None:
            self.bar.reset(total=len(ids))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Advance the bar at each test's teardown and show the failed count.

        Args:
            report: One phase's report of a test.
        """
        if report.failed:
            self.failed += 1
            self.bar.set_postfix(failed=self.failed)
        if report.when == "teardown":
            self.bar.update()


def _notebook_path(line: str) -> Path:
    """Find the notebook: the argument, else `JPY_SESSION_NAME`, else `__vsc_ipynb_file__`.

    Args:
        line: The argument, possibly empty.

    Returns:
        The notebook's path, an `.ipynb` or the `.py` jupytext opened as a notebook.

    Raises:
        RuntimeError: None of the three names a notebook.
    """
    user_ns = getattr(get_ipython(), "user_ns", {})
    found = line.strip() or os.environ.get("JPY_SESSION_NAME") or user_ns.get("__vsc_ipynb_file__")
    if not found:
        raise RuntimeError(
            "The notebook's path is unknown; pass it as an argument, such as "
            "`%%monte_carlo path/to/notebook.ipynb`."
        )
    return Path(found)


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
