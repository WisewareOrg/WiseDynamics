"""`FswHost`: a build ticked at its period, fed by role, and kept whole when it fails."""

from collections.abc import Callable, Mapping
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
    """A build that logs its boot and each tick, and commands nothing.

    It logs `"boot"`, then each tick's `t_ns`, in `events`; each tick's roles in `roles`; each
    tick's `t_ns` to the `imu` reading's `AngVelPlatform` in `gyro`; and each tick's payloads, as
    given, in `payloads`, and each tick's mapping itself in `mappings`. It raises `_Boom` in
    `fail_in`, `"boot"` or `"tick"`, and keeps it in `raised`. The `commanding` fixture's build
    commands.

    Attributes:
        period_s: The build's period, in seconds.
    """

    period_s = 0.5

    def __init__(self, fail_in: str | None = None):
        """Set where the build fails.

        Args:
            fail_in: `"boot"` or `"tick"` to raise `_Boom` there, else `None`.
        """
        self.events: list[str | int] = []
        self.gyro: dict[int, list[float]] = {}
        self.roles: list[list[str]] = []
        self.payloads: list[Mapping[str, Any]] = []
        self.mappings: list[Any] = []
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
            `{}`.
        """
        self.events.append(t_ns)
        self.roles.append(sorted(messages_by_role))
        self.payloads.append(dict(messages_by_role))
        self.mappings.append(messages_by_role)
        self.gyro[t_ns] = list(messages_by_role["imu"].AngVelPlatform)
        self._maybe_fail("tick")
        return {}

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
    scenario = Leo500(ImuSpacecraft())
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    scenario.run(host, {}, duration_s=2.0)

    assert host.period_s == build.period_s
    assert build.events == ["boot", *(k * HALF_SECOND_NS for k in range(1, 5))]
    assert host.failure is None


def test_the_vehicle_s_readings_reach_the_build_by_role_as_sampled_at_the_tick():
    """Each tick gets every role of `readers()`, and the IMU reading of that tick's instant."""
    scenario = Leo500(ImuSpacecraft())
    scenario.disperse_case(np.random.SeedSequence(0))
    build = _Stub()

    datasets = scenario.run(FswHost(build, scenario.vehicle), {}, duration_s=2.0)

    assert build.roles == [sorted(scenario.vehicle.readers())] * 4
    t_s, _ = datasets["t_s"]
    gyro_P, _ = datasets["imu/gyro_P"]
    for t_ns, reading in build.gyro.items():
        (row,) = np.flatnonzero(t_s == t_ns / 1e9)
        assert reading == gyro_P[row].tolist(), t_ns


def test_every_tick_gets_each_role_s_payload_taken_at_reset():
    """Each role's payload is the same object at every tick."""
    scenario = Leo500(ImuSpacecraft())
    build = _Stub()

    scenario.run(FswHost(build, scenario.vehicle), {}, duration_s=2.0)

    first, *later = build.payloads
    assert first and later
    for tick, payloads in enumerate(later, start=2):
        for role in first:
            assert payloads[role] is first[role], f"tick {tick} got a new {role} payload"


def test_a_tick_cannot_change_the_mapping_of_payloads_it_is_given():
    """The mapping every tick shares is read-only."""
    scenario = Leo500(ImuSpacecraft())
    build = _Stub()

    scenario.run(FswHost(build, scenario.vehicle), {}, duration_s=0.5)

    with pytest.raises(TypeError):
        build.mappings[0]["imu"] = None


@pytest.mark.parametrize("where", ["boot", "tick"])
def test_an_exception_in_the_build_surfaces_whole_from_run(where: str):
    """`run` raises the build's own exception object, its traceback reaching into the build.

    Args:
        where: Where the build raises.
    """
    scenario = Leo500(ImuSpacecraft())
    build = _Stub(fail_in=where)
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(_Boom, match=f"failed in {where}") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert raised.value is build.raised
    assert host.failure is build.raised
    assert raised.traceback[-1].name == "_maybe_fail"
    assert raised.traceback[-2].name == where
    assert raised.value.__suppress_context__


def test_a_command_for_a_role_with_no_writer_fails_the_run_at_that_tick(
    commanding: Callable[..., Any],
):
    """A tick commanding a role with no writer fails, naming it; the build is not ticked again.

    Args:
        commanding: The stub build that commands what it is given.
    """
    scenario = Leo500(ImuSpacecraft())
    build = commanding({"wheel": 0.1})
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(ValueError, match="wheel") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.ticked == 1


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
    scenario = Leo500(ImuSpacecraft())
    host = FswHost(_Stub(), scenario.vehicle) if hosted else None

    with pytest.raises(RuntimeError, match="SWIG director method error"):
        if hosted:
            scenario.run(host, {}, duration_s=2.0, ground=_Raising())
        else:
            scenario.run(_Raising(), {}, duration_s=2.0)

    if hosted:
        assert host is not None
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
    scenario = Leo500(_Unsubscribed())
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(RuntimeError, match=r"roles \['spare'\] are not subscribed to any message") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.events == []


def test_a_role_never_written_by_the_first_tick_fails_it_before_the_build_ticks():
    """`run` raises the host's error naming the unwritten role and the first tick's time."""
    scenario = Leo500(_Unwritten())
    build = _Stub()
    host = FswHost(build, scenario.vehicle)

    with pytest.raises(RuntimeError, match=r"roles \['spare'\] have not been written at t = 0.5 s") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert build.events == ["boot"]


class _Commanded(ImuSpacecraft):
    """An `ImuSpacecraft` with one writer, role `spare`, a stand-alone `ArrayMotorTorqueMsg`."""

    def __init__(self, **kwargs: Any) -> None:
        """Build the vehicle and the stand-alone message its writer is.

        Args:
            **kwargs: As `ImuSpacecraft` takes them.
        """
        super().__init__(**kwargs)
        self.spare_cmd = messaging.ArrayMotorTorqueMsg()

    def writers(self) -> dict[str, Any]:
        """Hand out the vehicle's writers.

        Returns:
            Role `spare`: the stand-alone message `spare_cmd`.
        """
        return {"spare": self.spare_cmd}


def test_a_command_is_written_to_its_role_s_writer_at_the_tick_by_the_host(
    commanding: Callable[..., Any], torque: Callable[[float], Any]
):
    """Each tick's command is in its writer's message from that tick's instant, by the host.

    Args:
        commanding: The stub build that commands what it is given.
        torque: Builds a wheel torque command.
    """
    vehicle = _Commanded()
    scenario = Leo500(vehicle)
    host = FswHost(commanding({"spare": torque(0.25)}), vehicle)
    command = vehicle.spare_cmd
    reader = messaging.ArrayMotorTorqueMsgReader()
    reader.subscribeTo(command)

    datasets = scenario.run(host, {"cmd": (command, "motorTorque", "N*m")}, duration_s=2.0)

    t_s, _ = datasets["t_s"]
    written, _ = datasets["cmd"]
    assert written[t_s == 0.0, 0].tolist() == [0.0]
    assert set(written[t_s >= 0.5, 0]) == {0.25}
    assert reader.timeWritten() == 4 * HALF_SECOND_NS
    assert reader.moduleID() == host.moduleID


def test_a_command_set_naming_a_role_with_no_writer_is_refused_whole_before_any_write(
    commanding: Callable[..., Any], torque: Callable[[float], Any]
):
    """Commanding roles with no writer fails, naming them and the writer roles; none is written.

    Args:
        commanding: The stub build that commands what it is given.
        torque: Builds a wheel torque command.
    """
    vehicle = _Commanded()
    scenario = Leo500(vehicle)
    commands = {"spare": torque(0.25), "wheel": torque(0.5), "arm": torque(1.0)}
    host = FswHost(commanding(commands), vehicle)
    reader = messaging.ArrayMotorTorqueMsgReader()
    reader.subscribeTo(vehicle.spare_cmd)

    with pytest.raises(
        ValueError,
        match=r"roles \['arm', 'wheel'\], which have no writer; the writer roles are \['spare'\]",
    ) as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
    assert not reader.isWritten()


def test_a_role_absent_from_a_tick_keeps_its_last_command(
    commanding: Callable[..., Any], torque: Callable[[float], Any]
):
    """Commanded at the first tick only, the writer's message holds that command to the end.

    Args:
        commanding: The stub build that commands what it is given.
        torque: Builds a wheel torque command.
    """
    vehicle = _Commanded()
    scenario = Leo500(vehicle)
    build = commanding({"spare": torque(0.25)}, ticks=1)
    command = vehicle.spare_cmd

    datasets = scenario.run(FswHost(build, vehicle), {"cmd": (command, "motorTorque", "N*m")}, duration_s=2.0)

    t_s, _ = datasets["t_s"]
    written, _ = datasets["cmd"]
    assert build.ticked == 4
    assert set(written[t_s >= 0.5, 0]) == {0.25}


@pytest.mark.parametrize(
    "payload",
    [0.25, messaging.IMUSensorMsgPayload()],
    ids=["not-a-payload", "another-message-s-payload"],
)
def test_a_command_its_writer_rejects_fails_the_run_kept_whole(commanding: Callable[..., Any], payload: Any):
    """The writer's own error fails the tick, kept whole in `failure` and raised from `run`.

    Args:
        commanding: The stub build that commands what it is given.
        payload: A command the `ArrayMotorTorqueMsg` writer cannot take.
    """
    scenario = Leo500(_Commanded())
    host = FswHost(commanding({"spare": payload}), scenario.vehicle)

    with pytest.raises(TypeError, match="ArrayMotorTorqueMsgPayload") as raised:
        scenario.run(host, {}, duration_s=2.0)

    assert host.failure is raised.value
