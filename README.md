# WiseDynamics

WiseDynamics is a thin wrapper over Basilisk
([ADR 0001 rev 1](docs/decisions/0001-basilisk-simulation-host.md)). It provides reusable
vehicles and prebuilt scenarios, each scenario an operating environment that builds and places
its vehicle. A pytest plugin exposes them to a consumer's tests: for each case it builds a
scenario, with its vehicle, from the case's own random streams, and the case writes one HDF5
run file. WiseDynamics holds no flight software and does no analysis of the run files.

Vehicles live in `wise_dynamics/basilisk/vehicles/` and scenarios in
`wise_dynamics/basilisk/scenarios/`, where `SCENARIOS` in the package's `__init__.py` registers
each scenario by name. A scenario names its vehicle class. A consumer picks a scenario with
`--sim-scenario`, reaches its vehicle as `scenario.vehicle`, and reads the parameters of either
from its class attributes and class docstring.

## Using it

A consumer depends on a tag:

```toml
[project.optional-dependencies]
sim = ["wise-dynamics"]

[tool.uv.sources]
wise-dynamics = { git = "https://github.com/WisewareOrg/WiseDynamics", tag = "v0.1.0" }
```

Installing it registers the pytest plugin. A Monte Carlo run is the consumer's test, parametrized
over `--sim-runs` cases in one pytest session; `-n` runs workers in parallel and needs
`pytest-xdist`, which the consumer adds itself:

```sh
pytest tests/sim -n 8 --sim-runs 852 --sim-out DIR
```

[`wise_dynamics/basilisk/README.md`](wise_dynamics/basilisk/README.md) is the reference: the
modules, what a vehicle and a scenario are, the plugin's options, fixtures and case seeding, and a
worked example.

WiseDynamics depends on no flight software; flight software depends on WiseDynamics.

## Layout

- `wise_dynamics/` — the installed package: the pytest plugin, and the `basilisk` subpackage below.
- `wise_dynamics/basilisk/` — the vehicle and scenario interfaces, the vehicles and the scenarios.
- `tests/` — the test suite.
- `docs/decisions/` — this repository's ADRs.

[CONTRIBUTING.md](CONTRIBUTING.md) covers how a change is made and checked.
