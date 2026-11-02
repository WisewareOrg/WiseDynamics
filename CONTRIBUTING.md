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

A PR whose base is not the default branch needs the Development-field link written by hand — see
WiseKiosk's [`link-pr` skill](https://github.com/WisewareOrg/WiseKiosk/blob/main/.claude/skills/link-pr/SKILL.md)
for the GraphQL mutation; `Closes #N` in the body only writes that record on a default-base PR.

`checks.yml` runs on every pull request and on `main`:

- `check-adrs` — `docs/decisions/` and its index agree, and every citation of an ADR in the
  tracked tree pins that ADR's current rev.
- `check-sim-independent` — `scripts/check-sim-independent.sh`; the root README's Independence
  section states what it enforces.
- `ruff` — `uv run ruff check .` and `uv run ruff format --check .`.
- `pytest` — `uv run pytest`, the full suite.

The same commands run locally after `uv sync`. A change that moves the golden reference
regenerates it with `uv run python -m tests.test_golden_trajectory` and says why in its pull
request.
