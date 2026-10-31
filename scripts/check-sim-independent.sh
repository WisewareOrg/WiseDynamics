#!/usr/bin/env bash
set -euo pipefail

# sim/ and tests/ depend on no flight software (ADR 0001 rev 2): a match here means a
# flight-software package leaked in, including inside an import statement. Substring matching,
# not word-bounded, so a match inside a longer identifier (e.g. gnc_cdh_transport) still counts.
matches=$(command grep -rniE --include='*.py' \
  'gnc_algorithms|gnc_cdh|gnc_analysis|wise_gnc' \
  sim tests || true)

if [ -n "$matches" ]; then
  echo "$matches"
  exit 1
fi
