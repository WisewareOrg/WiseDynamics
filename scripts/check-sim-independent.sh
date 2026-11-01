#!/usr/bin/env bash
set -euo pipefail

# sim/ and tests/ depend on no flight software (ADR 0001 rev 2): a match here means a
# flight-software package leaked in, including inside an import statement or a declared
# dependency, under any name the gnc-* family might take (gnc_algorithms, gnc_cdh, gnc_analysis,
# wise_gnc, and any future sibling). Substring matching, not word-bounded, so a match inside a
# longer identifier (e.g. gnc_cdh_transport) still counts. Python modules use underscores
# (gnc_cdh); PyPI distribution names in pyproject.toml use hyphens (gnc-cdh).
code_matches=$(command grep -rniE --include='*.py' 'gnc_[a-z]|wise_gnc' sim tests || true)
dep_matches=$(command grep -niE 'gnc-[a-z]' pyproject.toml || true)

if [ -n "$code_matches" ] || [ -n "$dep_matches" ]; then
  [ -n "$code_matches" ] && echo "$code_matches"
  [ -n "$dep_matches" ] && echo "$dep_matches"
  exit 1
fi
