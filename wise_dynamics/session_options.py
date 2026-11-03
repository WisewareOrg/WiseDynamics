"""A pytest session's parsed `--sim-*` options."""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SessionOptions"]


@dataclass(frozen=True)
class SessionOptions:
    """A pytest session's `--sim-*` options, parsed and checked once by the plugin.

    Attributes:
        plant: `--sim-spacecraft`, a name in `wise_dynamics.basilisk.PLANTS`.
        seed: `--sim-seed`, the session seed, non-negative.
        runs: `--sim-runs`, the number of cases, at least 1.
        duration_s: `--sim-duration-s`, each case's simulated duration in seconds, positive.
        out_dir: The absolute directory the run files go to.

    Raises:
        ValueError: `seed`, `runs` or `duration_s` is out of range.
    """

    plant: str
    seed: int
    runs: int
    duration_s: float
    out_dir: Path

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"--sim-seed must be non-negative, got {self.seed}")
        if self.runs < 1:
            raise ValueError(f"--sim-runs must be at least 1, got {self.runs}")
        if not self.duration_s > 0:
            raise ValueError(f"--sim-duration-s must be positive, got {self.duration_s}")
