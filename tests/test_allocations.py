"""The allocation helpers: they count a path's per-tick native allocations, and only that path's.

The IMU path they are tested on allocates twice a tick in Basilisk's `Saturate::saturate` and nowhere
else, as of the bsk this repository locks; a bsk that changes that changes these counts.
"""

import os
import signal
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from Basilisk.fswAlgorithms import inertial3D

from tests import allocations
from tests.allocations import (
    _name,
    _owner,
    _tracked,
    assert_no_native_allocations_per_tick,
    native_allocations,
    native_allocations_by_function,
    native_allocations_per_tick,
)
from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


def _fly(ticks: int) -> int:
    """Fly Leo500 with the IMU spacecraft for `ticks` one-second ticks.

    Args:
        ticks: The number of ticks to fly.

    Returns:
        How many times the IMU was updated.
    """
    vehicle = ImuSpacecraft()
    Leo500(vehicle).run(_Guidance(), {}, duration_s=float(ticks))
    return vehicle.imu.CallCounts


def test_a_path_that_allocates_every_tick_counts_each_allocation(tmp_path: Path):
    """The IMU's update allocates twice a tick.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert native_allocations_per_tick(_fly, within="ImuSensor::UpdateState", ticks=50, directory=tmp_path) == 2.0


def test_a_path_that_ran_and_allocates_nothing_per_tick_holds(tmp_path: Path):
    """The IMU's update less the saturation it calls ran and allocates nothing per tick.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert_no_native_allocations_per_tick(
        _fly, within="ImuSensor::UpdateState", excluding=["Saturate::saturate"], ticks=50, directory=tmp_path
    )


def test_a_path_that_allocates_every_tick_does_not_hold(tmp_path: Path):
    """The IMU's update, saturation included, fails the hold.

    Args:
        tmp_path: The test's own temporary directory.
    """
    with pytest.raises(AssertionError, match="makes 2.0 native allocations per tick"):
        assert_no_native_allocations_per_tick(_fly, within="ImuSensor::UpdateState", ticks=50, directory=tmp_path)


def test_a_path_that_never_ran_does_not_hold(tmp_path: Path):
    """A flight reporting that its path never ran fails the hold, though nothing allocated.

    Args:
        tmp_path: The test's own temporary directory.
    """

    def fly_without_the_path(ticks: int) -> int:
        """Fly, and report that the path never ran.

        Args:
            ticks: The number of ticks to fly.

        Returns:
            0.
        """
        _fly(ticks)
        return 0

    with pytest.raises(AssertionError, match="never ran"):
        assert_no_native_allocations_per_tick(
            fly_without_the_path, within="no such function", ticks=10, directory=tmp_path
        )


def test_each_allocation_is_charged_to_the_function_that_made_it(tmp_path: Path):
    """The IMU's two allocations a tick are charged to `Saturate::saturate`, not to the Eigen that carries them.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert native_allocations_by_function(_fly, ticks=50, directory=tmp_path)["Saturate::saturate"] == 2.0


def test_a_tracked_flight_runs_in_another_process(tmp_path: Path):
    """The flight memray tracks runs in a child process, so this one is never tracked.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert _tracked(lambda ticks: os.getpid(), 1, tmp_path / "capture.bin") != os.getpid()


def _raises(ticks: int) -> None:
    """Raise, as a flight that fails does.

    Args:
        ticks: The number of ticks the flight was to fly.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"a flight of {ticks} ticks failed")


@pytest.mark.parametrize(
    ("fly", "reason"),
    [
        (_raises, "exited with 1"),
        (lambda ticks: object(), "exited with 1"),
        (lambda ticks: os.kill(os.getpid(), signal.SIGKILL), f"was killed by signal {int(signal.SIGKILL)}"),
    ],
    ids=["raises", "returns no JSON", "killed"],
)
def test_a_tracked_flight_that_fails_in_its_child_says_how(fly: Callable[[int], object], reason: str, tmp_path: Path):
    """A flight that raises, returns what JSON cannot carry, or is killed fails the capture, saying how.

    Args:
        fly: The failing flight.
        reason: How the error says the child ended.
        tmp_path: The test's own temporary directory.
    """
    with pytest.raises(RuntimeError, match=reason):
        _tracked(fly, 1, tmp_path / "capture.bin")


def test_a_bare_string_of_exclusions_is_refused(tmp_path: Path):
    """`excluding` takes function names, so a single name passed bare, which reads as its characters, raises.

    Args:
        tmp_path: The test's own temporary directory.
    """
    with pytest.raises(TypeError, match="not the string"):
        native_allocations_per_tick(
            _fly, within="ImuSensor::UpdateState", excluding="Saturate::saturate", ticks=5, directory=tmp_path
        )


def test_a_path_is_named_by_its_whole_function_name(tmp_path: Path):
    """A function name counts the allocations under it; part of one counts none.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert native_allocations(_fly, within="ImuSensor::UpdateState", ticks=5, directory=tmp_path) > 0
    assert native_allocations(_fly, within="ImuSensor::Update", ticks=5, directory=tmp_path) == 0


def test_a_name_no_allocation_has_on_its_stack_does_not_hold(tmp_path: Path):
    """A held path whose name no allocation's stack holds fails, though the path ran and nothing allocated.

    Args:
        tmp_path: The test's own temporary directory.
    """
    with pytest.raises(AssertionError, match="the name is wrong"):
        assert_no_native_allocations_per_tick(_fly, within="ImuSensor::Update", ticks=10, directory=tmp_path)


@pytest.mark.parametrize(
    ("resolved", "name"),
    [
        ("Saturate::saturate(Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const>&)", "Saturate::saturate"),
        ("(anonymous namespace)::RungeKutta::integrate(double, double)", "RungeKutta::integrate"),
        ("new_threadstate.llvm.387420898227119684", "new_threadstate"),
        ("pthread_getattr_np@@GLIBC_2.32", "pthread_getattr_np"),
        ("atexit_register.cold", "atexit_register"),
    ],
)
def test_a_frame_is_named_without_its_parameters_namespace_or_build_suffixes(resolved: str, name: str):
    """A frame's name drops what differs between builds of one library.

    Args:
        resolved: The function as memray resolves it.
        name: The name it is known by.
    """
    assert _name(resolved) == name


@pytest.mark.parametrize(
    ("functions", "owner"),
    [
        (
            ["operator new", "std::vector<double>::_M_realloc_insert", "Recorder<X>::UpdateState"],
            "Recorder<X>::UpdateState",
        ),
        (
            ["Eigen::internal::aligned_malloc", "HubEffector::computeHubOnlyDerivatives"],
            "HubEffector::computeHubOnlyDerivatives",
        ),
        (["operator new", "<unknown>", "ExtendedStateVector::map"], "ExtendedStateVector::map"),
        (["operator new", "std::allocator<int>::allocate"], "<unknown>"),
    ],
    ids=["standard library", "Eigen", "unnamed frame", "all plumbing"],
)
def test_an_allocation_is_charged_to_its_innermost_named_frame_past_the_plumbing(functions: list[str], owner: str):
    """The allocator's, the standard library's, Eigen's and unnamed frames are passed over.

    Args:
        functions: An allocation's function names, innermost first.
        owner: The function it is charged to.
    """
    assert _owner(functions) == owner


def test_a_tracked_flight_returns_a_result_larger_than_a_pipe_holds(tmp_path: Path):
    """A result larger than the pipe's buffer reaches this process whole.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert _tracked(lambda ticks: "x" * 200_000, 1, tmp_path / "capture.bin") == "x" * 200_000


def test_a_tracked_flight_that_runs_past_its_time_is_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A flight still running at the deadline is killed, and the capture fails saying so.

    Args:
        tmp_path: The test's own temporary directory.
        monkeypatch: Shortens the deadline.
    """
    monkeypatch.setattr(allocations, "_FLIGHT_TIMEOUT_S", 0.5)
    with pytest.raises(RuntimeError, match="ran past 0.5 s and was killed"):
        _tracked(lambda ticks: time.sleep(30), 1, tmp_path / "capture.bin")
