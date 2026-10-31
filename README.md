# WiseDynamics

WiseDynamics is a reusable Basilisk plant library for Wiseware's GNC work. It depends on no
flight software: a consumer that hosts a named algorithm against one of its plants pins a
tagged release of this repository, rather than this repository depending on that consumer.
Every simulation plant is a sibling directory under `sim/` — Basilisk first (`sim/basilisk/`),
with other hosts such as Trick (`sim/trick/`) joining later as siblings rather than
replacements.

Each host directory provides static plant models, a one-case runner, a run writer for the HDF5
layout a consumer's analysis reads, and a pytest plugin exposing Monte Carlo options and
fixtures (`--sim-runs`/`--sim-out`/`--sim-seed`/`--sim-duration-s`/`--sim-spacecraft`;
`plant`/`draws`/`run_writer`). Installing the package registers the plugin with pytest
automatically, via its own `pytest11` entry point -- a consumer's `pytest_plugins` must not also
name it. A consumer builds its own flight-software module against the `plant` fixture's
`readers()`/`initial_state()`/`mu`, and owns its own scenario and campaign tests. A campaign is
that consumer's own test run at scale: `pytest <its scenario path> --sim-runs N --sim-out DIR
[-n WORKERS]`.

## Layout

- `sim/basilisk/` — the Basilisk simulation host, the first plant
  ([ADR 0001 rev 2](docs/decisions/0001-basilisk-first-simulation-host.md)); see its own `README.md`.
- `sim/<host>/` — a future simulation host, as a sibling directory.
- `docs/decisions/` — this repository's ADRs.
