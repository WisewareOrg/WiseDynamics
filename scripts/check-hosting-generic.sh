#!/usr/bin/env bash
set -euo pipefail

# sim/basilisk/hosting/ hosts any WiseGncDesign algorithm generically; a match here means
# algorithm-specific vocabulary leaked into the generic host.
matches=$(command grep -rniE --include='*.py' \
  'imu|gyro|accel|mrp|sigma|reckon|navatt|navtrans|star|tracker' \
  sim/basilisk/hosting/ || true)

if [ -n "$matches" ]; then
  echo "$matches"
  exit 1
fi
