# WiseDynamics

WiseDynamics is a thin wrapper over Basilisk
([ADR 0001 rev 1](docs/decisions/0001-basilisk-simulation-host.md)). It provides reusable
vehicles and prebuilt scenarios, each scenario an operating environment that builds and places
its vehicle. A pytest plugin exposes them to a consumer's tests: for each case it builds a
scenario, with its vehicle, from the case's own random streams, and the case writes one HDF5
run file. WiseDynamics holds no flight software and does no analysis of the run files.

Vehicles live in `wise_dynamics/basilisk/vehicles/` and scenarios in
`wise_dynamics/basilisk/scenarios/`, one module each, registered by name in `SCENARIOS`. A scenario
names its vehicle class, which the `sim_vehicle` mark replaces. A consumer picks a scenario with
the `sim_scenario` mark, reaches its vehicle as `scenario.vehicle`, and reads the parameters of either from its class attributes and
class docstring.

## Using it

A consumer depends on a tag:

```toml
[project.optional-dependencies]
sim = ["wise-dynamics"]

[tool.uv.sources]
wise-dynamics = { git = "https://github.com/WisewareOrg/WiseDynamics", tag = "v0.1.0" }
```

Installing it registers the pytest plugin. Each consumer test that uses a `sim_` fixture is one
Monte Carlo campaign, configured by marks (see the basilisk README); `-n` runs workers in parallel
and needs `pytest-xdist`, which the consumer adds itself:

```sh
pytest --sim-session first
pytest tests/sim -n 8 --full-monte-carlo
```

Without `--full-monte-carlo` each test flies at most two cases, into the `scratch` session.

Run files are kept in named sessions, which the basilisk README documents; a consumer adds the
one-line `just session [name]` recipe that this repository's `justfile` has.

The `notebook` extra adds a `%%monte_carlo` magic and `load_runs`; see the basilisk README.

[`wise_dynamics/basilisk/README.md`](wise_dynamics/basilisk/README.md) is the reference: the
modules, what a vehicle and a scenario are, the plugin's marks, fixtures and case seeding, and a
worked example.

WiseDynamics depends on no flight software; flight software depends on WiseDynamics.

## Layout

- `wise_dynamics/` — the installed package: the pytest plugin and its sessions, and the `basilisk` subpackage below.
- `wise_dynamics/basilisk/` — the vehicle and scenario interfaces, the vehicles and the scenarios.
- `tests/` — the test suite.
- `justfile` — the `session` recipe.
- `docs/decisions/` — this repository's ADRs.

[CONTRIBUTING.md](CONTRIBUTING.md) covers how a change is made and checked.
