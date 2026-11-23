"""Count the native allocations a named code path makes per simulation tick, with memray."""

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import memray

__all__ = [
    "assert_no_native_allocations_per_tick",
    "native_allocations_by_function",
    "native_allocations_per_tick",
]

_RELEASES = {memray.AllocatorType.FREE, memray.AllocatorType.MUNMAP, memray.AllocatorType.PYMALLOC_FREE}
_PLUMBING = ("malloc", "calloc", "realloc", "memalign", "mmap", "operator new", "std::", "__gnu_cxx", "Eigen::")
"""Substrings of the frames that only carry an allocation to the allocator, which never own one."""


def _name(function: str) -> str:
    """A native frame's function name without its parameter list or anonymous namespace.

    Args:
        function: The frame's function, as memray resolves it.

    Returns:
        The name.
    """
    return function.replace("(anonymous namespace)::", "").split("(")[0]


def _stacks(fly: Callable[[int], object], ticks: int, file: Path) -> list[tuple[list[str], int]]:
    """Fly `ticks` ticks under memray and give each native allocation's stack, innermost frame first.

    One untracked flight of `ticks` ticks goes first, so imports and lazy set-up land outside the
    capture.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        file: The memray capture file to write and remove, which must not exist.

    Returns:
        One `(function names, allocation count)` per allocation record.
    """
    fly(ticks)
    with memray.Tracker(file, native_traces=True, trace_python_allocators=False):
        fly(ticks)
    stacks = [
        ([_name(function) for function, _, _ in record.native_stack_trace()], record.n_allocations)
        for record in memray.FileReader(file).get_allocation_records()
        if memray.AllocatorType(record.allocator) not in _RELEASES
    ]
    file.unlink()
    return stacks


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
    excluded = tuple(excluding)
    short = _count(_stacks(fly, ticks, directory / "short.bin"), within, excluded)
    long = _count(_stacks(fly, 2 * ticks, directory / "long.bin"), within, excluded)
    return (long - short) / ticks


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
    ran; `fly` therefore returns how many times the path ran, which the caller counts.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks, and returns how many
            times the path ran.
        within: A substring of the native function name that names the path.
        excluding: Substrings of native function names the path calls but does not own.
        ticks: The number of ticks of the shorter flight.
        directory: Where the memray capture files are written and removed.

    Raises:
        AssertionError: A flight reported that the path never ran, or the path allocates per tick.
    """
    runs: list[int] = []

    def counted(n: int) -> None:
        """Fly `n` ticks and keep how many times the path ran.

        Args:
            n: The number of ticks to fly.
        """
        runs.append(fly(n))

    per_tick = native_allocations_per_tick(
        counted, within=within, excluding=excluding, ticks=ticks, directory=directory
    )
    if not all(runs):
        raise AssertionError(f"the path {within!r} never ran in a flight: it ran {runs} times")
    if per_tick != 0.0:
        raise AssertionError(f"the path {within!r} makes {per_tick} native allocations per tick")


def native_allocations_by_function(fly: Callable[[int], object], *, ticks: int, directory: Path) -> dict[str, float]:
    """Every native allocation per tick, by the function that made it.

    An allocation belongs to the innermost frame of its stack that is not allocator plumbing
    (`malloc`, `operator new`, the standard library, Eigen), so an Eigen temporary is charged to the
    function that built it. The slope is taken as `native_allocations_per_tick` takes it.

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
            owner = next((f for f in functions if not any(p in f for p in _PLUMBING)), "<unknown>")
            owned[owner] += count
        return owned

    short = owners(_stacks(fly, ticks, directory / "short.bin"))
    long = owners(_stacks(fly, 2 * ticks, directory / "long.bin"))
    slopes = {function: (long[function] - short[function]) / ticks for function in long | short}
    return dict(sorted(((f, s) for f, s in slopes.items() if s), key=lambda item: -item[1]))
