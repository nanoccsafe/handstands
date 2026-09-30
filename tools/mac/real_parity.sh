#!/usr/bin/env bash
# Real-clip parity from Linux: N real clips -> fixtures -> the Mac ->
# RealParityTests — the check chainlink #42 calls the one that counts.
#
#   tools/mac/real_parity.sh [N] [remote-dir]
#
# N           how many real clips to check (default 20)
# remote-dir  directory on the Mac that receives swift/ (default wt/real-parity)
#
# What it does, from the repo root on Linux:
#
#   1. cd pipeline && uv run python -m handstand.golden --real-sample N — the
#      first N clips (sorted by clip_id) that have athlete keypoints AND at
#      least one hold, written to <data_dir>/golden_real/<clip_id>.json.
#      That tree is git-ignored, and golden refuses anything under swift/,
#      ios/, pipeline/ or docs/ outright: real keypoints never enter the
#      repo. The repo's synthetic Fixtures/golden/parity_reference.json is
#      copied beside the fixtures so the directory is self-contained.
#   2. rsync <data_dir>/golden_real/ to $host:~/handstand-private/golden_real/
#      — a private folder OUTSIDE ~/wt and ~/GitRepo, created if missing and
#      never deleted from (rsync runs without --delete, and the mkdir only
#      ever touches that one directory).
#   3. swift_sync — the function tools/mac/swift_test.sh shares from
#      tools/mac/swift_sync.sh — rsyncs swift/, then on the Mac:
#          HANDSTAND_REAL_GOLDEN=~/handstand-private/golden_real \
#              swift test --filter RealParityTests
#      inside swift/HandstandCore, and the summary lines are printed —
#      including "real parity: X/Y clips identical".
#   4. Exits non-zero if any clip mismatches (RealParityTests fails) or if
#      any step before it failed.
#
# Never touches ~/GitRepo on the Mac and never uses sudo.
# Env: HANDSTAND_MAC_HOST (default macmini).
set -euo pipefail

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "real_parity: $*" >&2; exit 1; }

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

clips="${1:-20}"
remote_dir="${2:-wt/real-parity}"
[[ "$clips" =~ ^[1-9][0-9]*$ ]] || die "clip count must be a positive integer, got '$clips'"
host="${HANDSTAND_MAC_HOST:-macmini}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/tools/mac/swift_sync.sh"

reference="$repo_root/swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden/parity_reference.json"
[[ -f "$reference" ]] || die "missing $reference"
# Under $HOME on the Mac, outside ~/wt and ~/GitRepo: private, never committed.
private_dir="handstand-private/golden_real"

# ------------------------------------------------------------------ 1. fixtures
echo "== golden: the first $clips real clip(s) that hold"
(cd "$repo_root/pipeline" && uv run python -m handstand.golden --real-sample "$clips") \
    || die "handstand.golden --real-sample $clips failed"

data_dir="$(
    cd "$repo_root/pipeline" &&
        uv run python -c 'from handstand.paths import data_dir; print(data_dir())'
)" || die "cannot resolve the data directory"
real_dir="$data_dir/golden_real"
[[ -d "$real_dir" ]] || die "no fixtures: $real_dir does not exist"
count="$(find "$real_dir" -maxdepth 1 -name '*.json' ! -name 'parity_reference.json' | wc -l)"
[[ "$count" -ge 1 ]] || die "no real fixtures in $real_dir — every candidate was skipped?"
# Beside them, so the directory is self-contained on the Mac (it is synthetic:
# it is the reference the fixtures' own expected.score was scored with).
cp "$reference" "$real_dir/" || die "cannot copy the parity reference into $real_dir"
echo "== $count real fixture(s) in $real_dir (parity reference copied beside them)"

# ------------------------------------------------------- 2. private data over
echo "== rsync fixtures to $host:$private_dir/"
ssh "$host" "mkdir -p ~/$private_dir" || die "cannot create ~/$private_dir on $host"
rsync -a "$real_dir/" "$host:~/$private_dir/" || die "rsync of $real_dir failed"

# ------------------------------------------------------- 3. swift, then the run
swift_sync "$host" "$remote_dir" "$repo_root" || die "rsync of swift/ failed"

tmp_dir="${TMPDIR:-/tmp/opencode}"
mkdir -p "$tmp_dir"
log="$(mktemp "$tmp_dir/real_parity.XXXXXX")"
trap 'rm -f "$log"' EXIT

echo "== swift test --filter RealParityTests on $host: $remote_dir/swift/HandstandCore"
failed=0
ssh "$host" "cd '$remote_dir/swift/HandstandCore' && \
    export HANDSTAND_REAL_GOLDEN=\"\$HOME/$private_dir\" && \
    swift build && swift test --filter RealParityTests" 2>&1 | tee "$log" || failed=1

# The lines the report quotes: the test's own summary, plus XCTest's.
echo "== summary"
summary="$(grep -E "real parity: |Executed [0-9]+ test|Test Suite 'All tests'" "$log" || true)"
if [[ -z "$summary" ]]; then
    echo "  (no summary line found in the output above)"
else
    while IFS= read -r line; do echo "  $line"; done <<< "$summary"
fi

if [[ "$failed" -ne 0 ]]; then
    die "real parity FAILED on $host in $remote_dir/swift/HandstandCore"
fi
