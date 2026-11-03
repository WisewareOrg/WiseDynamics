"""One field of one recorded message, with its units."""

from dataclasses import dataclass

__all__ = ["RecordedField"]


@dataclass(frozen=True)
class RecordedField:
    """One field of one recorded message, and the units it is expressed in.

    Attributes:
        role: The message's role, a key of `Plant.truth_messages()` or of the flight software's
            outputs handed to `run_case`.
        field: The payload field the recorder exposes, for example `"r_BN_N"`.
        units: The units string a run file stores with it, for example `"m"`.
    """

    role: str
    field: str
    units: str
