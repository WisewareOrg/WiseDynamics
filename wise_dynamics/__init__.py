"""WiseDynamics: spacecraft plant models whose Monte Carlo cases run as pytest tests.

This package holds the standard-dispersion draw and the run-file writer;
`wise_dynamics.pytest_plugin` is the pytest plugin, and `wise_dynamics.basilisk` holds the
Basilisk plants and runner. `wise_dynamics/basilisk/README.md` is the reference.
"""

from wise_dynamics.dispersion import draw_standard
from wise_dynamics.run_file import write_run

__all__ = ["draw_standard", "write_run"]
