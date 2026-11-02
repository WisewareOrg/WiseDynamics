# WiseDynamics

WiseDynamics is a library of spacecraft plant models for Wiseware's GNC work, with a one-case
runner, an HDF5 run writer and a pytest plugin for Monte Carlo campaigns. Flight software pins a
tagged release and runs its own scenario tests against these plants. Basilisk is the first
simulation host ([ADR 0001 rev 1](docs/decisions/0001-basilisk-first-simulation-host.md)).

## Using it

A consumer depends on a tag:

```toml
[project.optional-dependencies]
sim = ["wise-dynamics"]

[tool.uv.sources]
wise-dynamics = { git = "https://github.com/WisewareOrg/WiseDynamics", tag = "v0.1.0" }
```

Installing it registers the pytest plugin. A campaign is the consumer's scenario test at scale:

```sh
pytest tests/sim -n 8 --sim-runs 852 --sim-out DIR
```

[`wise_dynamics/basilisk/README.md`](wise_dynamics/basilisk/README.md) is the reference: the
plants and their parameters, a worked one-case example, campaigns, run files, and the plugin's
options and fixtures.

## Independence

WiseDynamics depends on Basilisk, numpy, h5py and pytest. Flight software depends on
WiseDynamics, and the dependency runs that one way: `scripts/check-sim-independent.sh` fails on a
flight-software package named in `wise_dynamics/`, `tests/` or `pyproject.toml`.

## Layout

- `wise_dynamics/` — the installed package: campaigns, dispersion, run files, the pytest plugin.
- `wise_dynamics/basilisk/` — the Basilisk host: the plant contract, the plants, the runner.
  A further simulation host is a sibling subpackage.
- `tests/` — the regression suite that gates a Basilisk upgrade, including the golden reference
  `tests/golden/`.
- `scripts/` — CI gates.
- `docs/decisions/` — this repository's ADRs.

[CONTRIBUTING.md](CONTRIBUTING.md) covers how a change is made and checked.
