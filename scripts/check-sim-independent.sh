#!/usr/bin/env bash
set -euo pipefail

# Exits 1 and prints each match when wise_dynamics/ or tests/ names a flight-software package
# in Python source (an identifier containing gnc_<letter> or wise_gnc), or pyproject.toml names
# a gnc-<letter> distribution. The README's "Independence" section states what this enforces.
code_matches=$(command grep -rniE --include='*.py' 'gnc_[a-z]|wise_gnc' wise_dynamics tests || true)
dep_matches=$(command grep -niE 'gnc-[a-z]' pyproject.toml || true)

if [ -n "$code_matches" ] || [ -n "$dep_matches" ]; then
  [ -n "$code_matches" ] && echo "$code_matches"
  [ -n "$dep_matches" ] && echo "$dep_matches"
  exit 1
fi
