"""WiseDynamics: spacecraft plant models and Monte Carlo campaigns of them.

This package holds what a campaign needs from any simulator: `Campaign` and `Case`, the
`Dispersion` of a configuration, and the run-file writer. `wise_dynamics.basilisk` holds the
Basilisk plants and runner; its `README.md` is the reference for both.
"""

from wise_dynamics.campaign import Campaign, Case, distribution_version
from wise_dynamics.dispersion import Dispersion, apply_draws
from wise_dynamics.run_file import Dataset, write_run

__all__ = [
    "Campaign",
    "Case",
    "Dataset",
    "Dispersion",
    "apply_draws",
    "distribution_version",
    "write_run",
]
