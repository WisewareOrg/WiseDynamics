# 0004 — Augment Basilisk with compiled modules built against its SDK, delivered through the git pin

**Status:** accepted
**Decided:** 2026-11-23 (issue 57, compiled Runge-Kutta)
**Rev:** 1

## Revisions

- **rev 1** — 2026-11-23 — first written (issue 57, compiled Runge-Kutta).

## Context

[ADR 0001 rev 1](0001-basilisk-simulation-host.md) makes Basilisk the simulation host, installed from its
wheels. Some of what a case spends on the simulation side is not physics but the way stock Basilisk
code is built: Basilisk's Runge-Kutta integrator rebuilds string-keyed maps of heap matrices at every
stage, and dominated a case's native allocations and its dynamics. Nothing in WiseDynamics' Python can
reach inside a Basilisk C++ module, and the Basilisk wheel cannot be changed from here.

Whatever replaces stock Basilisk code has to keep every run file bit-identical, or it changes the data
every analysis was judged on, and has to reach a consumer, which takes WiseDynamics by a git pin and
nothing else.

## Decision

WiseDynamics augments Basilisk with C++ modules of its own where stock Basilisk code costs more than its
work needs, such as the explicit Runge-Kutta integrator of the `lo` and `med` fidelities.

- **Built against Basilisk's SDK, at install.** A module is compiled when wise-dynamics is installed,
  against the headers of `bsk-sdk`, Basilisk's own package for building against its wheels, and links
  the installed wheel's libraries. A consumer's git pin is all it takes: no index, no wheel of ours.
- **One exact Basilisk, moved by the tests.** A module reads Basilisk's classes directly, so it is correct
  only against the release it was compiled for. `bsk` and `bsk-sdk` are pinned to one release, in
  `dependencies` and in `build-system.requires`, and move together as one Renovate update; a minor or patch
  update merges itself when the build and the tests pass, a major one waits for review, and a consumer
  takes either with its next WiseDynamics pin.
- **Refused, not trusted, on a mismatch.** Importing a module refuses a Basilisk whose source revision, or
  whose recorded build facts, differ from the module's: a layout mismatch corrupts memory rather than
  raising, and another instruction set changes the arithmetic.
- **Held to the code it replaces.** A module does the floating-point operations of the Basilisk code it
  replaces, in the same order, so a run is bit-identical; and it allocates nothing per step of its own.
  Both are tests in this repository, against stock Basilisk in the same process.

## Alternatives considered

- **Stock Basilisk only.** Keeps WiseDynamics pure Python, and leaves every case paying for Basilisk's
  integrator.
- **Patching Basilisk and building it from source.** [ADR 0001 rev 1](0001-basilisk-simulation-host.md)
  installs Basilisk from its wheels and never builds it; a Basilisk built on every install, or a wheel of
  our own to host, costs far more than one module.
- **Changing Basilisk itself.** The cost stays in every case until a Basilisk release carries the change,
  on a schedule this repository does not set.
- **Basilisk's headers vendored in this repository.** `bsk-sdk` ships the same headers and Eigen with each
  Basilisk release and checks the version at configure; a vendored copy would be re-copied by hand at
  every release.
- **Prebuilt wheels of wise-dynamics on a package index.** A consumer would have to add the index, a change
  to how wise-dynamics is consumed.
- **A Basilisk range, the module compiled at first import against whatever is installed.** No pin, and a
  compiler, a cache and a lock against parallel first imports in every consumer's process; the pin moved by
  the tests gives the same hands-off upgrade with none of those.
- **A Basilisk range, falling back to stock Basilisk on a mismatch.** A consumer never breaks, and silently
  loses the module's gain behind a warning, with two integration paths to keep bit-identical.
- **A simulator of our own in place of Basilisk's dynamics.** Gives up bit-identical runs and the host
  [ADR 0001 rev 1](0001-basilisk-simulation-host.md) chose, for physics that is already compiled.

## Consequences

Installing wise-dynamics compiles its modules, so every place it is installed needs a C++17 compiler. It
builds for Linux x86_64 only, the platform whose parity with Basilisk is tested, with the arithmetic of
Basilisk's own build whatever the compiler defaults to; elsewhere the install fails at configure, saying so.

A consumer resolves exactly the Basilisk WiseDynamics pins and cannot move Basilisk ahead of it. A
Basilisk release that changes what a module reads fails the update's build or tests and does not merge;
it then waits for a WiseDynamics change.

A consumer's machine compiles its own modules and runs none of this repository's tests; what holds its
build to the tested arithmetic is that the build flags are fixed, and that a module whose build facts
differ from Basilisk's is refused at import.

`hi` keeps Basilisk's adaptive RKF78, which no module replaces. The integrator costs that
[ADR 0003 rev 2](0003-fidelity-tiers.md) weighs are those of Basilisk's own integrators, which `lo` and
`med` no longer run; its choice of method for each fidelity rests on the accuracy budget, unchanged.
