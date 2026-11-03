# WiseDynamics

WiseDynamics is a library of spacecraft plant models for Wiseware's GNC work, with a one-case
runner, an HDF5 run writer and a pytest plugin that runs each Monte Carlo case as a pytest test.
Flight software pins a tagged release and runs its own scenario tests against these plants.
Basilisk is the first simulation host
([ADR 0001 rev 1](docs/decisions/0001-basilisk-first-simulation-host.md)).

## Using it

A consumer depends on a tag:

```toml
[project.optional-dependencies]
sim = ["wise-dynamics"]

[tool.uv.sources]
wise-dynamics = { git = "https://github.com/WisewareOrg/WiseDynamics", tag = "v0.1.0" }
```

Installing it registers the pytest plugin. A Monte Carlo run is the consumer's scenario test,
parametrized over `--sim-runs` cases in one pytest session:

```sh
pytest tests/sim -n 8 --sim-runs 852 --sim-out DIR
```

[`wise_dynamics/basilisk/README.md`](wise_dynamics/basilisk/README.md) is the reference: the
plants and their parameters, a worked one-case example, run files, and the plugin's options,
fixtures and case seeding.

WiseDynamics depends on no flight software; flight software depends on WiseDynamics.

## Layout

- `wise_dynamics/` — the installed package: dispersion, run files, the pytest plugin.
- `wise_dynamics/basilisk/` — the Basilisk host: the plant interface, the plants, the runner.
  A further simulation host is a sibling subpackage.
- `tests/` — the test suite.
- `docs/decisions/` — this repository's ADRs.

[CONTRIBUTING.md](CONTRIBUTING.md) covers how a change is made and checked.
