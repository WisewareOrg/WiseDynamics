"""`FswHost`: a build ticked at its period, fed by role, and kept whole when it fails."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel

from wise_dynamics.basilisk import FswHost, ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

HALF_SECOND_NS = 500_000_000


class _Boom(Exception):
    """The stub build's own exception."""


class _Stub:
    """A build that logs its boot and each tick, and returns what it is told to.

    It logs `"boot"`, then each tick's `t_ns`, in `events`; each tick's roles in `roles`; and each
    tick's `t_ns` to the `imu` reading's `AngVelPlatform` in `gyro`. Each tick returns `commands`.
    It raises `_Boom` in `fail_in`, `"boot"` or `"tick"`, and keeps it in `raised`.

    Attributes:
        period_s: The build's period, in seconds.
    """

    period_s = 0.5

    def __init__(self, commands: Mapping[str, Any] | None = None, fail_in: str | None = None):
        """Set what the build returns and where it fails.

        Args:
            commands: What each tick returns; `{}` when not given.
            fail_in: `"boot"` or `"tick"` to raise `_Boom` there, else `None`.
        """
        self.events: list[str | int] = []
        self.gyro: dict[int, list[float]] = {}
        self.roles: list[list[str]] = []
        self.commands = {} if commands is None else commands
        self.fail_in = fail_in
        self.raised: _Boom | None = None

    def boot(self) -> None:
        """Log the boot, and raise when told to fail there."""
        self.events.append("boot")
        self._maybe_fail("boot")

    def tick(self, t_ns: int, messages_by_role: Mapping[str, Any]) -> Mapping[str, Any]:
        """Log the tick and its readings, and raise when told to fail there.

        Args:
            t_ns: The tick's simulation time, in nanoseconds.
            messages_by_role: The vehicle's readings by role.

        Returns:
            `commands`.
        """
        self.events.append(t_ns)
        self.roles.append(sorted(messages_by_role))
        self.gyro[t_ns] = list(messages_by_role["imu"].AngVelPlatform)
        self._maybe_fail("tick")
        return self.commands

    def _maybe_fail(self, where: str) -> None:
        """Raise `_Boom` when `where` is where the build is told to fail.

        Args:
            where: `"boot"` or `"tick"`.

        Raises:
            _Boom: When `where` is `fail_in`.
        """
        if self.fail_in == where:
            try:
                raise _Boom(f"failed in {where}")
            except _Boom as error:
                self.raised = error
                raise


def test_a_build_boots_then_ticks_at_its_period_from_one_period_in():
    """Boot comes first; the ticks follow at each period, from `period_s` to the end, inclusive."""
    scenario = Leo500()
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    scenario.run(host, {}, duration_s=2.0)

    assert host.period_s == build.period_s
    assert build.events == ["boot", *(k * HALF_SECOND_NS for k in range(1, 5))]
    assert host.failure is None


def test_the_vehicle_s_readings_reach_the_build_by_role_as_sampled_at_the_tick():
    """Each tick gets every role of `readers()`, and the IMU reading of that tick's instant."""
    scenario = Leo500()
    scenario.disperse(np.random.default_rng(0), noise_seed=1)
    build = _Stub()

    datasets = scenario.run(FswHost(build, scenario.vehicle), {}, duration_s=2.0)

    assert build.roles == [sorted(scenario.vehicle.readers())] * 4
    t_s, _ = datasets["t_s"]
    gyro_P, _ = datasets["imu/gyro_P"]
    for t_ns, reading in build.gyro.items():
        (row,) = np.flatnonzero(t_s == t_ns / 1e9)
        assert reading == gyro_P[row].tolist(), t_ns


@pytest.mark.parametrize("where", ["boot", "tick"])
def test_an_exception_in_the_build_surfaces_whole_from_run(where: str):
    """`run` raises the build's own exception object, its traceback reaching into the build.

    Args:
        where: Where the build raises.
    """
    scenario = Leo500()
    build = _Stub(fail_in=where)
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(_Boom, match=f"failed in {where}") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert raised.value is build.raised
    assert host.failure is build.raised
    assert raised.traceback[-1].name == "_maybe_fail"
    assert raised.traceback[-2].name == where
    assert raised.value.__suppress_context__


def test_a_command_for_a_role_with_no_writer_fails_the_run_at_that_tick():
    """A tick commanding a role with no writer fails, naming it; the build is not ticked again."""
    scenario = Leo500()
    build = _Stub(commands={"wheel": 0.1})
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(ValueError, match="wheel") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.events == ["boot", HALF_SECOND_NS]


class _Raising(sysModel.SysModel):
    """A module that raises `_Boom` at its first update after t = 0.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 0.5

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Raise `_Boom` after t = 0.

        Args:
            CurrentSimNanos: The current simulation time, in nanoseconds.

        Raises:
            _Boom: At the first update after t = 0.
        """
        if CurrentSimNanos > 0:
            raise _Boom("not the host's")


@pytest.mark.parametrize("hosted", [False, True], ids=["plain-fsw-module", "host-with-no-failure"])
def test_an_error_no_host_kept_passes_from_run_as_basilisk_raised_it(hosted: bool):
    """With no kept failure, `run` raises Basilisk's own error, not one the host stored.

    The raising module is the flight-software module itself, or the ground beside an `FswHost`.

    Args:
        hosted: Whether the flight software is an `FswHost`, with the raising module as ground.
    """
    scenario = Leo500()
    host = FswHost(_Stub(), scenario.vehicle) if hosted else None

    with pytest.raises(RuntimeError, match="SWIG director method error"):
        if hosted:
            scenario.run(host, {}, duration_s=2.0, ground=_Raising())
        else:
            scenario.run(_Raising(), {}, duration_s=2.0)

    if hosted:
        assert host.failure is None


class _Unsubscribed(ImuSpacecraft):
    """An `ImuSpacecraft` with one more reader, `spare`, subscribed to nothing."""

    def readers(self) -> dict[str, Any]:
        """Build the vehicle's readers and an unsubscribed `spare`.

        Returns:
            Role `imu` as `ImuSpacecraft` gives it, and role `spare`, subscribed to nothing.
        """
        return {**super().readers(), "spare": messaging.IMUSensorMsgReader()}


class _Unwritten(ImuSpacecraft):
    """An `ImuSpacecraft` with one more reader, `spare`, subscribed to a message never written."""

    def readers(self) -> dict[str, Any]:
        """Build the vehicle's readers and a `spare` whose message is never written.

        Returns:
            Role `imu` as `ImuSpacecraft` gives it, and role `spare`, subscribed to a stand-alone
            message kept on the vehicle and never written.
        """
        self.spare_msg = messaging.IMUSensorMsg()
        spare = messaging.IMUSensorMsgReader()
        spare.subscribeTo(self.spare_msg)
        return {**super().readers(), "spare": spare}


def test_a_role_subscribed_to_nothing_fails_the_reset_before_boot():
    """`run` raises the host's error naming the unsubscribed role; the build never boots."""
    scenario = Leo500(vehicle_type=_Unsubscribed)
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(
        RuntimeError, match=r"roles \['spare'\] are not subscribed to any message"
    ) as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.events == []


def test_a_role_never_written_by_the_first_tick_fails_it_before_the_build_ticks():
    """`run` raises the host's error naming the unwritten role and the first tick's time."""
    scenario = Leo500(vehicle_type=_Unwritten)
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(
        RuntimeError, match=r"roles \['spare'\] have not been written at t = 0.5 s"
    ) as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.events == ["boot"]
