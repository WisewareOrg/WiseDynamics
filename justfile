# Make the named Monte Carlo session active, or list the sessions when no name is given.
session name="":
    uv run --no-sync pytest --sim-session {{name}}
