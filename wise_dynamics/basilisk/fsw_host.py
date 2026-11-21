"""The Basilisk module hosting any flight-software build in a scenario's flight-software slot."""

from collections.abc import Mapping
from typing import Any, Protocol

from Basilisk.architecture import sysModel

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["FswBuild", "FswHost"]


class FswBuild(Protocol):
    """What a flight-software build provides to be hosted.

    Attributes:
        period_s: The build's period, in seconds: the host ticks it at this period.
    """

    period_s: float

    def boot(self) -> None:
        """Start the build, before its first tick."""

    def tick(self, t_ns: int, messages_by_role: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one period of the build.

        Args:
            t_ns: The simulation time, in Basilisk's integer nanoseconds.
            messages_by_role: The vehicle's readings, role to the message payload its reader
                gives, unconverted. Each payload is valid only during this tick, a view of its
                reader's buffer that the next read overwrites, so copy what you keep.

        Returns:
            Actuator commands by role, `{}` when it commands nothing: each a payload of the
            message type the role's writer carries. A role absent from a tick keeps its last
            command, since its message keeps its last payload.
        """
        ...


class FswHost(sysModel.SysModel):
    """Hosts one flight-software build: boots it at reset, then ticks it at its period.

    `Reset` calls `build.boot()` and takes the reset time as the last tick. Each `UpdateState`
    whose time lies after the last tick reads every reader of the vehicle once, calls
    `build.tick(t_ns, {role: payload})` and writes each returned command to its role's writer at
    that time, so the device reads it before the next dynamics step. The first tick is one period
    after the reset; the call Basilisk makes at the reset time ticks nothing. A tick commanding any
    role with no writer fails before any of its commands is written. A command its writer rejects,
    such as a payload of another message, fails the tick with the writer's own error; which of the
    tick's other commands were written is not said, since the run has failed. A reader subscribed
    to no message fails the reset before `boot`, and one whose message is unwritten at the first
    tick fails that tick before `tick`. An exception from `boot` or `tick`, from a write, or from
    either check, is kept whole in `failure` and re-raised, and `Scenario.run` raises that same
    exception in place of Basilisk's director error.

    Attributes:
        failure: The exception `boot`, a tick, a write or a reader check raised, kept whole, else
            `None`.
    """

    failure: BaseException | None

    def __init__(self, build: FswBuild, vehicle: Vehicle) -> None:
        """Take the build and the vehicle's readers.

        Args:
            build: The build to host.
            vehicle: The vehicle whose `readers()` feed the build and whose `writers()` take its
                commands, each taken once.
        """
        super().__init__()
        self._build = build
        self._readers = vehicle.readers()
        self._writers = vehicle.writers()
        self._t_ns = 0
        self._inputs_written = False
        self.failure = None

    @property
    def period_s(self) -> float:
        """The build's period, in seconds: the period of the task the scenario runs the host in."""
        return self._build.period_s

    def Reset(self, CurrentSimNanos: int) -> None:
        """Check every reader is subscribed, boot the build; take the reset time as the last tick.

        Args:
            CurrentSimNanos: The reset time, in nanoseconds, as Basilisk passes it.

        Raises:
            RuntimeError: Naming the roles whose readers are subscribed to no message, before
                `boot`, kept whole in `failure`.
            BaseException: Whatever `boot` raised, kept whole in `failure`.
        """
        self._t_ns = CurrentSimNanos
        try:
            unlinked = sorted(
                role for role, reader in self._readers.items() if not reader.isLinked()
            )
            if unlinked:
                raise RuntimeError(f"roles {unlinked} are not subscribed to any message")
            self._build.boot()
        except BaseException as error:
            self.failure = error
            raise

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Tick the build with every reading by role and write its commands, after the last tick.

        Args:
            CurrentSimNanos: The current simulation time, in nanoseconds, as Basilisk passes it.

        Raises:
            RuntimeError: At the first tick, naming the roles whose messages have not been
                written, before `tick`, kept whole in `failure`.
            ValueError: Naming the roles the tick commanded that have no writer, and the writer
                roles, before any command is written, kept whole in `failure`.
            BaseException: Whatever `tick` or a writer raised, kept whole in `failure`.
        """
        if CurrentSimNanos <= self._t_ns:
            return
        self._t_ns = CurrentSimNanos
        try:
            if not self._inputs_written:
                unwritten = sorted(
                    role for role, reader in self._readers.items() if not reader.isWritten()
                )
                if unwritten:
                    t = CurrentSimNanos / 1e9
                    raise RuntimeError(f"roles {unwritten} have not been written at t = {t} s")
                self._inputs_written = True
            messages = {role: reader() for role, reader in self._readers.items()}
            commands = self._build.tick(CurrentSimNanos, messages)
            unwritable = sorted(set(commands) - set(self._writers))
            if unwritable:
                raise ValueError(
                    f"the build commanded roles {unwritable}, which have no writer; "
                    f"the writer roles are {sorted(self._writers)}"
                )
            for role, payload in commands.items():
                self._writers[role].write(payload, CurrentSimNanos, self.moduleID)
        except BaseException as error:
            self.failure = error
            raise
