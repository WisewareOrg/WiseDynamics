# 0002 — Keep Monte Carlo runs in named sessions

**Status:** accepted
**Decided:** 2026-11-09 (issue 22, sim sessions)
**Rev:** 1

## Revisions

- **rev 1** — 2026-11-09 — first written (issue 22, sim sessions).

## Context

Each pytest session wrote a new stamp folder under a test's directory, and `load_runs` read the
newest. A session interrupted partway was still the newest folder, so a reader could take a handful
of cases for the campaign, and rerunning one case wrote a further folder that hid the full set. Runs
reach the project through the pytest command line, the editor's test explorer and a notebook's
magic, and these meet only in the plugin; whatever fixes where runs go has to live there.

## Decision

The plugin owns sessions. A session is a named working set holding one set of run files per test,
and one project-wide root holds every session. Which session is active is stored in a pointer file
in the root, which the plugin and `load_runs` both read, so every way of running or reading cases
agrees on where runs are. There is no environment variable.

A case replaces its own result, and nothing else is cleared: a case that starts deletes its own
earlier run file and crash record. A case that fails without having written its run file leaves a
crash record. Every case in a session is therefore a run, a crash record or missing, and missing
means only that it never ran.

## Alternatives considered

- **Stamp folders read as "newest" or "newest finished".** The newest folder is the interrupted
  one, and "finished" needs a way to tell, which grows into the next alternative.
- **A completion marker.** A second piece of state to write, to keep consistent with the run files
  and to lose on a crash, to recover a distinction a single working set does not have.
- **A `reruns/` folder.** Moves the ambiguity to a second place and leaves a reader to choose
  between the full set and the reruns.
- **An environment variable.** A running notebook kernel and an editor's test explorer do not see
  a change made after they started, so the writer and the reader can disagree.
- **Consumer-owned sessions through `-o sim_out`.** Each consumer then reimplements the
  convention, and the editor and the notebook bypass any recipe that sets the option.
- **A root-resolution hook for consumers.** Hands every consumer the job of agreeing with the
  plugin about where the root is, which is what the pointer file settles once.
- **Clearing a test's folder before a full rerun.** Cannot tell a full rerun from a selected few,
  and destroys results the user did not ask to replace.
- **A standalone session command.** A second entry point that has to resolve the root the way the
  pytest ini lookup does, and can drift from it.
- **No crash record.** A crash then reads as a case that never ran, and with replacement a case
  that passed once and then crashes would still show its old passing run.

## Consequences

- A rerun of one case by its node id changes that case's files and nothing else.
- Deleting a session is removing its folder.
- A consumer that judges a session reads it through the plugin's own reader rather than the file
  layout.
