"""WiseDynamics: spacecraft plant models for Monte Carlo campaigns run as pytest sessions.

This package holds the `Dispersion` of a plant's configuration, the run-file writer and the
installed-version lookup a run file records; `wise_dynamics.pytest_plugin` is the pytest plugin.
`wise_dynamics.basilisk` holds the Basilisk plants and runner; its `README.md` is the reference.
"""

from wise_dynamics.dispersion import DataclassConfig, Dispersion
from wise_dynamics.provenance import distribution_version
from wise_dynamics.run_file import Dataset, write_run

__all__ = ["DataclassConfig", "Dataset", "Dispersion", "distribution_version", "write_run"]
