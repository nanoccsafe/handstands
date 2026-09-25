#!/usr/bin/env bash
# Show state of every worker log: running / done, plus the last lines.
# Usage: tools/worker/status.sh [lines]
lines="${1:-5}"
shopt -s nullglob
for log in /tmp/wt-i*.log; do
  name="$(basename "$log" .log)"
  if tail -n 1 "$log" | grep -q __DONE__; then state=DONE; else state=RUNNING; fi
  echo "== ${name#wt-} [$state]"
  grep -v __DONE__ "$log" | tail -n "$lines"
done
