# Contributing to WiseDynamics

WiseDynamics is public. Tickets, branches, pull-request linkage and pull-request titles follow
[WisewareOrg/.github's PROCESS.md](https://github.com/WisewareOrg/.github/blob/main/PROCESS.md),
not restated here.

## CI checks

Every pull request against the default branch runs `process-gates.yml`:

- `check-branch` — the branch is named `<type>_<number>-<name>` for an open, milestoned issue
  carrying exactly one type label, and the PR's Development field links that issue
  (PROCESS.md PROC-001–PROC-008, PROC-010).
- `pr-title` — the PR title is a Conventional Commits subject (PROC-009).
- `repo-floor` — the repository's rulesets meet [`repository-floor.yml`](https://github.com/WisewareOrg/.github/blob/main/repository-floor.yml)
  (PROC-011).
- `check-python-floor` — the root `pyproject.toml` meets the org Python floor (wise-ci).

A PR whose base is not the default branch needs the Development-field link written by hand — see
WiseKiosk's [`link-pr` skill](https://github.com/WisewareOrg/WiseKiosk/blob/main/.claude/skills/link-pr/SKILL.md)
for the GraphQL mutation; `Closes #N` in the body only writes that record on a default-base PR.

`checks.yml` runs on every pull request and on `main`:

- `check-adrs` — `docs/decisions/` and its index agree, and every citation of an ADR in the
  tracked tree pins that ADR's current rev.
- `check-python` — after `uv sync --locked --all-extras`, wise-ci's `check-python` action: ruff check
  and format, interrogate, pydoclint and ty. The action's tool versions are wise-ci's at the commit
  `uses:` names; the `check-python` dev dependency's `rev` in `pyproject.toml` names the same
  commit, so a local run sees the same versions. Renovate bumps the two together.
- `pytest` — `uv run pytest`, the full suite.

Locally, after `uv sync --all-extras`, run `uv run --locked check-python` and `uv run pytest`.

## Measuring a case's cost

`uv run python -m benchmarks.case_budget` times a warm Leo500 case layer by layer, from dynamics alone
to the run file, each layer flown through `Scenario.run`; `--allocations` adds each layer's native
allocations per tick and the functions that make them, counted with memray. It is run by hand, never in
CI, since wall time on a shared runner says nothing.

`tests/allocations.py` counts the native allocations per tick that a named code path makes, as the slope
between two flights. `assert_no_native_allocations_per_tick` holds a path to none; memray cannot see a
path that ran without allocating, so the flight it is given returns how many times the path ran, and
the hold fails when that is 0.
