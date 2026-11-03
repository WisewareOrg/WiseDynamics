"""WiseDynamics: spacecraft vehicle models whose Monte Carlo cases run as pytest tests.

This package holds the run file; `wise_dynamics.pytest_plugin` is the pytest plugin, and
`wise_dynamics.basilisk` holds the Basilisk vehicles and runner. `wise_dynamics/basilisk/README.md`
is the reference.
"""

from wise_dynamics.run_file import RunFile

__all__ = ["RunFile"]
