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
- `pytest` — the full suite, on wise-dynamics installed as a git pin of the commit under test, as a
  consumer installs it, so it tests the built package and its compiled integrator, not the checkout.

Locally, after `uv sync --all-extras`, run `uv run --locked check-python` and `uv run pytest`.
`uv sync` compiles the integrator in `native/`, so it needs a C++17 compiler on `PATH`.
