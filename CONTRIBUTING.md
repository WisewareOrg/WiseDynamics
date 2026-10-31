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

`checks.yml` additionally runs `check-adrs`: `docs/decisions/` and its index table agree, and
every citation of an ADR anywhere in the tracked tree pins that ADR's current rev.
