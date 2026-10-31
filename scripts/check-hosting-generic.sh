#!/usr/bin/env bash
set -euo pipefail

# sim/basilisk/hosting/ hosts any WiseGncDesign algorithm generically; a whole-word match here
# means algorithm-specific vocabulary leaked into the generic host.
matches=$(command grep -rniE --include='*.py' \
  '\b(imu|gyro|accel|mrp|sigma|reckon|navatt|navtrans|star|tracker|omega|quaternion|dcm_pb|attitude)\b' \
  sim/basilisk/hosting/ || true)

if [ -n "$matches" ]; then
  echo "$matches"
  exit 1
fi
