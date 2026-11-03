"""One array destined for a run file, with its units."""

from typing import Any, NamedTuple

__all__ = ["Dataset"]


class Dataset(NamedTuple):
    """One array destined for a run file, with the units it is expressed in.

    Attributes:
        data: The values; written as float64, in whatever shape they have (for example `(N,)`
            for a time vector or `(N, 3)` for a vector history).
        units: The units string stored as the dataset's `units` attribute, for example `"s"`,
            `"m"`, `"rad/s"` or `"MRP"`.
    """

    data: Any
    units: str
