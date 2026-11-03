"""Type aliases for the Basilisk message, reader, recorder and module objects, typed as `Any`."""

from typing import Any

__all__ = ["Message", "MessageReader", "Recorder", "SysModel"]

type Message = Any
"""A Basilisk output message, such as `SCStatesMsg`."""

type MessageReader = Any
"""A Basilisk message reader, such as `IMUSensorMsgReader`."""

type Recorder = Any
"""A Basilisk message recorder, as `message.recorder(period_ns)` returns."""

type SysModel = Any
"""A Basilisk module or recorder that runs in a task."""
