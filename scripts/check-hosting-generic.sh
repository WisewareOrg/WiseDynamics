#!/usr/bin/env bash
set -euo pipefail

# sim/basilisk/hosting/ hosts any WiseGncDesign algorithm generically; a match here means
# algorithm-specific vocabulary leaked into the generic host, including inside an identifier
# (snake_case or camelCase), which is how this vocabulary actually leaks into code. Substring
# matching, not word-bounded: "tracker" replaces "star" (the one term that collided with the
# unrelated word "start") rather than bounding every term, since bounding loses identifiers.
matches=$(command grep -rniE --include='*.py' \
  'imu|gyro|accel|mrp|sigma|reckon|navatt|navtrans|tracker|omega|quaternion|dcm_pb|attitude' \
  sim/basilisk/hosting/ || true)

if [ -n "$matches" ]; then
  echo "$matches"
  exit 1
fi
