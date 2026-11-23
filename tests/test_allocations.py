"""`native_allocations_per_tick`: it counts a path's per-tick native allocations, and only that path's."""

from pathlib import Path

from Basilisk.fswAlgorithms import inertial3D

from tests.allocations import native_allocations_per_tick
from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


def _fly(ticks: int) -> None:
    """Fly Leo500 with the IMU spacecraft for `ticks` one-second ticks.

    Args:
        ticks: The number of ticks to fly.
    """
    Leo500(ImuSpacecraft()).run(_Guidance(), {}, duration_s=float(ticks))


def test_a_path_that_allocates_every_tick_counts_each_allocation(tmp_path: Path):
    """The IMU's update allocates twice a tick, both in Basilisk's `Saturate::saturate`.

    Args:
        tmp_path: The test's own temporary directory.
    """
    assert native_allocations_per_tick(_fly, within="ImuSensor::UpdateState", ticks=50, directory=tmp_path) == 2.0


def test_a_path_whose_allocations_are_all_excluded_counts_zero(tmp_path: Path):
    """The IMU's update less the saturation it calls allocates nothing per tick.

    Args:
        tmp_path: The test's own temporary directory.
    """
    per_tick = native_allocations_per_tick(
        _fly, within="ImuSensor::UpdateState", excluding=["Saturate::saturate"], ticks=50, directory=tmp_path
    )
    assert per_tick == 0.0
