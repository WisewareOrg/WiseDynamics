"""WiseDynamics: spacecraft plant models and Monte Carlo campaigns of them.

This package holds Monte Carlo campaigns (`Campaign`, `Case`), the `Dispersion` of a plant's
configuration, and the run-file writer. `wise_dynamics.basilisk` holds the Basilisk plants and
runner; its `README.md` is the reference for both.
"""

from wise_dynamics.campaign import Campaign, Case
from wise_dynamics.dispersion import Dispersion
from wise_dynamics.provenance import distribution_version
from wise_dynamics.run_file import Dataset, write_run

__all__ = [
    "Campaign",
    "Case",
    "Dataset",
    "Dispersion",
    "distribution_version",
    "write_run",
]
