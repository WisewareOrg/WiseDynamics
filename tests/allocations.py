"""Count the native allocations a named code path makes per simulation tick, with memray.

CPython's object allocator is left out: its small objects, its large ones, which it takes from `malloc`, and
the arenas it carves small ones from. Python objects are not native allocations, and whether a flight needs
one more arena depends on the heap the flight starts from, not on the flight.
"""

import json
import os
import re
import select
import signal
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import memray

__all__ = [
    "assert_no_native_allocations_per_tick",
    "native_allocations_by_function",
]

_RELEASES = {memray.AllocatorType.FREE, memray.AllocatorType.MUNMAP, memray.AllocatorType.PYMALLOC_FREE}
_PLUMBING = ("operator new", "std::", "__gnu_cxx::", "Eigen::", "<unknown>")
"""Prefixes of the frames that never own an allocation: the allocator's own, the standard library's,
Eigen's, and the frames memray cannot name, so an allocation is charged to the nearest named caller."""

_PYTHON_OBJECT_ALLOCATOR = frozenset(
    {
        "_PyObject_Malloc",
        "_PyObject_Calloc",
        "_PyObject_Realloc",
        "_PyMem_ArenaAlloc",
        "_PyObject_ArenaMmap",
        "new_arena",
    }
)
"""CPython's object allocator, whose allocations are not counted wherever it is the owner."""

_FLIGHT_TIMEOUT_S = 600.0
"""How long a tracked flight's child process may run before it is killed and the flight fails."""

type Stacks = list[tuple[list[str], int]]
"""Each allocation record's function names, innermost first, and its allocation count."""


def _name(function: str) -> str:
    """A native frame's function name without its parameter list, anonymous namespace or build suffixes.

    The suffixes a compiler or linker adds to a symbol (`.llvm.<hash>`, `.cold`, `@GLIBC_<version>`) differ
    between builds of one library, so they are dropped.

    Args:
        function: The frame's function, as memray resolves it.

    Returns:
        The name.
    """
    name = function.replace("(anonymous namespace)::", "").split("(")[0]
    return re.sub(r"(\.(llvm\.\d+|cold|isra\.\d+|constprop\.\d+|part\.\d+))+$|@@?GLIBC_[\d.]+$", "", name)


def _owner(functions: list[str]) -> str:
    """The function an allocation is charged to: the innermost frame that is not allocator plumbing.

    Args:
        functions: The allocation's function names, innermost first.

    Returns:
        The owner, `<unknown>` when every frame is plumbing.
    """
    return next((function for function in functions if not function.startswith(_PLUMBING)), "<unknown>")


def _tracked(fly: Callable[[int], object], ticks: int, file: Path) -> object:
    """Fly `ticks` ticks under memray in a forked child process, which writes the capture and exits.

    This process is never tracked, so memray's allocator hooks never touch it; see
    `test_a_tracked_flight_runs_in_another_process`. The child holds only the forking thread, so `fly`
    must not depend on another thread of this process.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        file: The memray capture file the child writes, which must not exist.

    Returns:
        What `fly` returned, which must be JSON.

    Raises:
        RuntimeError: The child failed: it exited with an error, after printing its traceback; it was
            killed by a signal; or it ran past `_FLIGHT_TIMEOUT_S`, and was killed.
    """
    read, write = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        os.close(read)
        status = 1
        try:
            with memray.Tracker(file, native_traces=True, trace_python_allocators=False):
                result = fly(ticks)
            os.write(write, json.dumps(result).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(write)
    deadline = time.monotonic() + _FLIGHT_TIMEOUT_S
    chunks = []
    with os.fdopen(read, "rb", buffering=0) as pipe:
        while chunk := _read_before(pipe, deadline):
            chunks.append(chunk)
        if chunk is None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise RuntimeError(f"the tracked flight of {ticks} ticks ran past {_FLIGHT_TIMEOUT_S} s and was killed")
    payload = b"".join(chunks)
    exit_code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
    if exit_code != 0:
        how = f"was killed by signal {-exit_code}" if exit_code < 0 else f"exited with {exit_code}"
        raise RuntimeError(f"the tracked flight of {ticks} ticks {how} in its child process")
    return json.loads(payload)


def _read_before(pipe: Any, deadline: float) -> bytes | None:
    """Read what the pipe holds next, waiting for it no later than `deadline`.

    Args:
        pipe: The read end of a pipe, unbuffered.
        deadline: The `time.monotonic()` by which something must arrive.

    Returns:
        The bytes read, empty at the end of the pipe, or None when the deadline passed first.
    """
    ready, _, _ = select.select([pipe], [], [], max(0.0, deadline - time.monotonic()))
    return pipe.read(1 << 16) if ready else None


def _stacks(fly: Callable[[int], object], ticks: int, file: Path) -> tuple[Stacks, object]:
    """Fly `ticks` ticks under memray and give each native allocation's stack, innermost frame first.

    One untracked flight of `ticks` ticks goes first, so imports and lazy set-up land outside the
    capture.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        file: The memray capture file to write and remove, which must not exist.

    Returns:
        Each allocation record's function names and count, and what the tracked flight's `fly` returned.
    """
    fly(ticks)
    result = _tracked(fly, ticks, file)
    stacks = [
        ([_name(function) for function, _, _ in record.native_stack_trace()], record.n_allocations)
        for record in memray.FileReader(file).get_allocation_records()
        if memray.AllocatorType(record.allocator) not in _RELEASES
    ]
    file.unlink()
    return stacks, result


def _names(excluding: Iterable[str]) -> tuple[str, ...]:
    """The function names to exclude, refusing a bare string, which would read as its characters.

    Args:
        excluding: Function names.

    Returns:
        The names.

    Raises:
        TypeError: `excluding` is a string.
    """
    if isinstance(excluding, str):
        raise TypeError(f"excluding takes function names, not the string {excluding!r}")
    return tuple(excluding)


def _native(stacks: Stacks) -> Stacks:
    """The stacks whose allocation is not CPython's object allocator's.

    Args:
        stacks: Each allocation's function names and count.

    Returns:
        Those stacks, less every one `_PYTHON_OBJECT_ALLOCATOR` owns.
    """
    return [(functions, count) for functions, count in stacks if _owner(functions) not in _PYTHON_OBJECT_ALLOCATOR]


def _count(stacks: Stacks, within: str, excluding: tuple[str, ...]) -> int:
    """Count the allocations whose stack holds the function `within` and none of `excluding`.

    Args:
        stacks: Each allocation's function names, without parameter lists, and count.
        within: The function name the stack must hold.
        excluding: Function names the stack must not hold.

    Returns:
        The number of matching allocations.
    """
    return sum(
        count for functions, count in stacks if within in functions and not any(e in functions for e in excluding)
    )


def _per_tick(
    fly: Callable[[int], object], within: str, excluding: tuple[str, ...], ticks: int, directory: Path
) -> tuple[float, int, list[object]]:
    """The allocations per tick under `within` and outside `excluding`, with the evidence a path is named.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        within: The function name that names the path.
        excluding: Function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Returns:
        The allocations per tick; every allocation the longer flight made under `within`, excluded ones,
        set-up and Python objects included; and the shorter and the longer tracked flight's `fly` results.
    """
    short, short_result = _stacks(fly, ticks, directory / "short.bin")
    long, long_result = _stacks(fly, 2 * ticks, directory / "long.bin")
    per_tick = (_count(_native(long), within, excluding) - _count(_native(short), within, excluding)) / ticks
    return per_tick, _count(long, within, ()), [short_result, long_result]


def assert_no_native_allocations_per_tick(
    fly: Callable[[int], int],
    *,
    within: str,
    excluding: Iterable[str] = (),
    ticks: int,
    directory: Path,
) -> None:
    """Fail unless a code path ran, is named by `within`, and made no native allocation per tick.

    memray sees only allocations, so `fly` returns how many times the path ran, which the caller counts,
    and each tracked flight's count must be above 0. The path is shown to be named by `within` when at
    least one allocation of the longer flight, set-up or excluded ones included, has the function
    `within` on its stack.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks, and returns how many
            times the path ran, as an int.
        within: The function name that names the path.
        excluding: Function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Raises:
        AssertionError: A flight reported that the path never ran, no allocation has `within` on its
            stack, or the path allocates per tick.
    """
    per_tick, named, runs = _per_tick(fly, within, _names(excluding), ticks, directory)
    if not all(runs):
        raise AssertionError(f"the path {within!r} never ran in a flight: it ran {runs} times")
    if not named:
        raise AssertionError(
            f"no allocation has {within!r} on its stack: the name is wrong, or the path allocates nothing at all"
        )
    if per_tick != 0.0:
        raise AssertionError(f"the path {within!r} makes {per_tick} native allocations per tick")


def native_allocations_by_function(fly: Callable[[int], object], *, ticks: int, directory: Path) -> dict[str, float]:
    """Every native allocation per tick, by the function that made it.

    An allocation belongs to its `_owner`: the innermost frame of its stack that is not allocator plumbing
    (`operator new`, the standard library, Eigen) and that memray can name, so an Eigen temporary is
    charged to the function that built it, and an allocation in an unnamed lambda to the function that
    called it. The slope is taken as `_per_tick` takes it.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Returns:
        Function name to its allocations per tick, for each function with a nonzero slope, most first.
    """

    def owners(stacks: Stacks) -> Counter[str]:
        """Count each allocation against the function that owns it.

        Args:
            stacks: Each allocation's function names and count.

        Returns:
            Function name to its allocation count.
        """
        owned: Counter[str] = Counter()
        for functions, count in _native(stacks):
            owned[_owner(functions)] += count
        return owned

    short = owners(_stacks(fly, ticks, directory / "short.bin")[0])
    long = owners(_stacks(fly, 2 * ticks, directory / "long.bin")[0])
    slopes = {function: (long[function] - short[function]) / ticks for function in long | short}
    return dict(sorted(((f, s) for f, s in slopes.items() if s), key=lambda item: -item[1]))
