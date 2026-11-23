"""The allocation helpers: they count a path's per-tick native allocations, and only that path's.

The IMU path they are tested on allocates twice a tick in Basilisk's `Saturate::saturate` and nowhere
else, as of the bsk this repository locks; a bsk that changes that changes these counts.
"""

from decimal import Decimal
from pathlib import Path

import pytest
from Basilisk.fswAlgorithms import inertial3D

from tests.allocations import (
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


def test_a_capture_leaves_this_processs_allocator_as_it_was(tmp_path: Path):
    """Decimal arithmetic, which reallocates memory the interpreter allocated before, works after a capture.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert native_allocations(_fly, within="ImuSensor::UpdateState", ticks=5, directory=tmp_path) > 0
    assert Decimal.from_float(1e-9) != Decimal(repr(1e-9))
