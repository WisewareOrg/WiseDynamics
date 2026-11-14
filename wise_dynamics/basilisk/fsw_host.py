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
            Actuator commands by role, `{}` when it commands nothing.
        """
        ...


class FswHost(sysModel.SysModel):
    """Hosts one flight-software build: boots it at reset, then ticks it at its period.

    `Reset` calls `build.boot()` and takes the reset time as the last tick. Each `UpdateState`
    whose time lies after the last tick reads every reader of the vehicle once and calls
    `build.tick(t_ns, {role: payload})`, so the first tick is one period after the reset; the call
    Basilisk makes at the reset time ticks nothing. A command the tick returns for a role with no
    writer fails the tick; until a vehicle declares writers, any command fails. An exception from
    `boot` or `tick` is kept whole in `failure` and re-raised, and `Scenario.run` raises that same
    exception in place of Basilisk's director error.

    Attributes:
        failure: The exception `boot` or a tick raised, kept whole, else `None`.
    """

    failure: BaseException | None

    def __init__(self, build: FswBuild, vehicle: Vehicle) -> None:
        """Take the build and the vehicle's readers.

        Args:
            build: The build to host.
            vehicle: The vehicle whose `readers()` feed the build, taken once.
        """
        super().__init__()
        self._build = build
        self._readers = vehicle.readers()
        self._t_ns = 0
        self.failure = None

    @property
    def period_s(self) -> float:
        """The build's period, in seconds: the period of the task the scenario runs the host in."""
        return self._build.period_s

    def Reset(self, CurrentSimNanos: int) -> None:
        """Boot the build and take the reset time as the last tick.

        Args:
            CurrentSimNanos: The reset time, in nanoseconds, as Basilisk passes it.

        Raises:
            BaseException: Whatever `boot` raised, kept whole in `failure`.
        """
        self._t_ns = CurrentSimNanos
        try:
            self._build.boot()
        except BaseException as error:
            self.failure = error
            raise

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Tick the build with every reading by role, when the time lies after the last tick.

        Args:
            CurrentSimNanos: The current simulation time, in nanoseconds, as Basilisk passes it.

        Raises:
            ValueError: Naming the roles the tick commanded that have no writer, kept whole in
                `failure`.
            BaseException: Whatever `tick` raised, kept whole in `failure`.
        """
        if CurrentSimNanos <= self._t_ns:
            return
        self._t_ns = CurrentSimNanos
        try:
            messages = {role: reader() for role, reader in self._readers.items()}
            commands = self._build.tick(CurrentSimNanos, messages)
            commanded = sorted(commands)
            if commanded:
                raise ValueError(
                    f"the build commanded roles {commanded}, which have no writer; "
                    "the writer roles are []"
                )
        except BaseException as error:
            self.failure = error
            raise
