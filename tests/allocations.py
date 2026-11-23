"""Count the native allocations a named code path makes per simulation tick, with memray."""

import json
import os
import re
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import memray

__all__ = [
    "assert_no_native_allocations_per_tick",
    "native_allocations",
    "native_allocations_by_function",
    "native_allocations_per_tick",
]

_RELEASES = {memray.AllocatorType.FREE, memray.AllocatorType.MUNMAP, memray.AllocatorType.PYMALLOC_FREE}
_PLUMBING = ("operator new", "std::", "__gnu_cxx::", "Eigen::", "<unknown>")
"""Prefixes of the frames that never own an allocation: the allocator's own, the standard library's,
Eigen's, and the frames memray cannot name, so an allocation is charged to the nearest named caller."""


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


def _tracked(fly: Callable[[int], object], ticks: int, file: Path) -> object:
    """Fly `ticks` ticks under memray in a child process of this one, which writes the capture.

    memray does not restore the allocator of a statically linked CPython, such as uv's, once it stops
    tracking: memory the interpreter allocated before is then freed with another allocator, which
    corrupts the heap (`Decimal` arithmetic aborts on it, for one). A forked child carries the
    tracking and exits, so this process is never tracked.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        file: The memray capture file the child writes, which must not exist.

    Returns:
        What `fly` returned, which must be JSON.

    Raises:
        RuntimeError: The flight raised in the child, whose traceback it printed.
    """
    read, write = os.pipe()
    with warnings.catch_warnings():
        # The child only flies and exits, so threads of this process it does not inherit are harmless.
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
    with os.fdopen(read, "rb") as pipe:
        payload = pipe.read()
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"the tracked flight of {ticks} ticks raised in its child process")
    return json.loads(payload)


def _stacks(fly: Callable[[int], object], ticks: int, file: Path) -> tuple[list[tuple[list[str], int]], object]:
    """Fly `ticks` ticks under memray and give each native allocation's stack, innermost frame first.

    One untracked flight of `ticks` ticks goes first, so imports and lazy set-up land outside the
    capture.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        file: The memray capture file to write and remove, which must not exist.

    Returns:
        One `(function names, allocation count)` per allocation record, and what the tracked flight's
        `fly` returned.
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


def _count(stacks: list[tuple[list[str], int]], within: str, excluding: tuple[str, ...]) -> int:
    """Count the allocations whose stack holds a frame matching `within` and none matching `excluding`.

    Args:
        stacks: Each allocation's function names and count.
        within: A substring of a function name the stack must hold.
        excluding: Substrings of function names the stack must not hold.

    Returns:
        The number of matching allocations.
    """
    return sum(
        count
        for functions, count in stacks
        if any(within in f for f in functions) and not any(e in f for f in functions for e in excluding)
    )


def native_allocations(fly: Callable[[int], object], *, within: str, ticks: int, directory: Path) -> int:
    """All the native allocations one flight makes under a named frame, set-up included.

    A path holding no allocation per tick may still allocate once, as it sets up; a count above 0
    shows memray resolves the path's frame, which a stripped library would hide.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        within: A substring of the native function name that names the path.
        ticks: The number of ticks to fly.
        directory: Where the memray capture file is written and removed.

    Returns:
        The number of allocations.
    """
    return _count(_stacks(fly, ticks, directory / "flight.bin")[0], within, ())


def native_allocations_per_tick(
    fly: Callable[[int], object],
    *,
    within: str,
    excluding: Iterable[str] = (),
    ticks: int,
    directory: Path,
) -> float:
    """The native allocations per tick that a code path makes, net of its one-off setup and teardown.

    An allocation counts when its native stack holds a frame whose function name contains `within`
    and none whose function name contains any of `excluding`, so `excluding` carves out what the
    path calls but does not own; `within=""` counts every allocation. Python-object allocations from
    pymalloc are not native and are not counted. The path is flown for `ticks` and then for
    `2 * ticks` ticks, each from a fresh simulation, and the difference is divided by `ticks`, so
    the result is the per-tick slope, exact since the counts are deterministic. A path that never
    ran reads 0 here; `assert_no_native_allocations_per_tick` rules that out.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        within: A substring of the native function name that names the path.
        excluding: Substrings of native function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Returns:
        The allocations per tick.
    """
    return _per_tick(fly, within, tuple(excluding), ticks, directory)[0]


def _per_tick(
    fly: Callable[[int], object], within: str, excluding: tuple[str, ...], ticks: int, directory: Path
) -> tuple[float, list[object]]:
    """The allocations per tick under `within` and outside `excluding`, and what each tracked flight returned.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        within: A substring of the native function name that names the path.
        excluding: Substrings of native function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Returns:
        The allocations per tick, and the shorter and the longer tracked flight's `fly` results.
    """
    short, short_result = _stacks(fly, ticks, directory / "short.bin")
    long, long_result = _stacks(fly, 2 * ticks, directory / "long.bin")
    return (_count(long, within, excluding) - _count(short, within, excluding)) / ticks, [short_result, long_result]


def assert_no_native_allocations_per_tick(
    fly: Callable[[int], int],
    *,
    within: str,
    excluding: Iterable[str] = (),
    ticks: int,
    directory: Path,
) -> None:
    """Fail unless a code path ran and made no native allocation per tick.

    memray sees only allocations, so a path that ran without allocating looks like one that never
    ran; `fly` therefore returns how many times the path ran, which the caller counts, and each
    tracked flight's count must be above 0. That count shows the path ran, not that `within` names it:
    `within` is a substring, so it is a qualified name (`Class::method`) that no other frame contains,
    and a path that allocates as it sets up is shown to be named by `native_allocations` counting more
    than 0 under it.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks, and returns how many
            times the path ran, as an int.
        within: A substring of the native function name that names the path.
        excluding: Substrings of native function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Raises:
        AssertionError: A flight reported that the path never ran, or the path allocates per tick.
    """
    per_tick, runs = _per_tick(fly, within, tuple(excluding), ticks, directory)
    if not all(runs):
        raise AssertionError(f"the path {within!r} never ran in a flight: it ran {runs} times")
    if per_tick != 0.0:
        raise AssertionError(f"the path {within!r} makes {per_tick} native allocations per tick")


def native_allocations_by_function(fly: Callable[[int], object], *, ticks: int, directory: Path) -> dict[str, float]:
    """Every native allocation per tick, by the function that made it.

    An allocation belongs to the innermost frame of its stack that is not allocator plumbing
    (`operator new`, the standard library, Eigen) and that memray can name, so an Eigen temporary is
    charged to the function that built it, and an allocation in an unnamed lambda to the function
    that called it. The slope is taken as `native_allocations_per_tick` takes it.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Returns:
        Function name to its allocations per tick, for each function with a nonzero slope, most first.
    """

    def owners(stacks: list[tuple[list[str], int]]) -> Counter[str]:
        """Count each allocation against the function that owns it.

        Args:
            stacks: Each allocation's function names and count.

        Returns:
            Function name to its allocation count.
        """
        owned: Counter[str] = Counter()
        for functions, count in stacks:
            owner = next((f for f in functions if not f.startswith(_PLUMBING)), "<unknown>")
            owned[owner] += count
        return owned

    short = owners(_stacks(fly, ticks, directory / "short.bin")[0])
    long = owners(_stacks(fly, 2 * ticks, directory / "long.bin")[0])
    slopes = {function: (long[function] - short[function]) / ticks for function in long | short}
    return dict(sorted(((f, s) for f, s in slopes.items() if s), key=lambda item: -item[1]))
