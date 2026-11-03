"""WiseDynamics: spacecraft plant models whose Monte Carlo cases run as pytest tests.

This package holds the run-file writer; `wise_dynamics.pytest_plugin` is the pytest plugin, and
`wise_dynamics.basilisk` holds the Basilisk plants and runner. `wise_dynamics/basilisk/README.md`
is the reference.
"""

from wise_dynamics.run_file import write_run

__all__ = ["write_run"]
