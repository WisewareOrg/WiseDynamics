"""Count the native allocations a named code path makes per simulation tick, with memray."""

from collections.abc import Callable, Iterable
from pathlib import Path

import memray
from memray._memray import AllocatorType

__all__ = ["native_allocations_per_tick"]

_RELEASES = {AllocatorType.FREE, AllocatorType.MUNMAP, AllocatorType.PYMALLOC_FREE}


def _count(fly: Callable[[int], None], ticks: int, within: str, excluding: Iterable[str], file: Path) -> int:
    """Count the native allocations made within `within` and outside `excluding` while flying `ticks` ticks.

    Args:
        fly: Builds and runs a fresh simulation for the given number of ticks.
        ticks: The number of ticks to fly.
        within: A substring of a native frame's function name the allocation's stack must contain.
        excluding: Substrings of native frame function names the allocation's stack must not contain.
        file: The memray capture file to write, which must not exist.

    Returns:
        The number of matching allocations.
    """
    with memray.Tracker(file, native_traces=True, trace_python_allocators=False):
        fly(ticks)
    count = 0
    for record in memray.FileReader(file).get_allocation_records():
        if AllocatorType(record.allocator) in _RELEASES:
            continue
        functions = [function for function, _, _ in record.native_stack_trace()]
        if any(within in f for f in functions) and not any(e in f for f in functions for e in excluding):
            count += record.n_allocations
    file.unlink()
    return count


def native_allocations_per_tick(
    fly: Callable[[int], None],
    *,
    within: str,
    excluding: Iterable[str] = (),
    ticks: int,
    directory: Path,
) -> float:
    """The native allocations per tick that a code path makes, net of its one-off setup and teardown.

    An allocation counts when its native stack holds a frame whose function name contains `within`
    and none whose function name contains any of `excluding`, so `excluding` carves out what the
    path calls but does not own. Python-object allocations from pymalloc are not native and are not
    counted. The path is flown for `ticks` and then for `2 * ticks` ticks, each from a fresh
    simulation, and the difference is divided by `ticks`, so the result is the per-tick slope. A
    path that allocates nothing per tick gives exactly 0, since the counts are deterministic.
    That a path that never ran also gives 0 is the caller's to rule out.

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
    short = _count(fly, ticks, within, excluded, directory / "short.bin")
    long = _count(fly, 2 * ticks, within, excluded, directory / "long.bin")
    return (long - short) / ticks
