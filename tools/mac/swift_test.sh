#!/usr/bin/env bash
# Build and test a Swift package of this repo on the Mac mini, from Linux.
#
#   tools/mac/swift_test.sh [package] [remote-dir]
#
# package     Package under swift/ to build and test (default HandstandCore)
# remote-dir  Directory on the Mac that receives swift/ (default wt/i38)
#
# The whole swift/ directory is rsynced to $host:$remote-dir/swift/ — never
# touching the remote .build or .swiftpm, so the build cache survives — and
# then `swift build && swift test` runs inside the package there. The test
# summary lines are printed at the end; any failure (rsync, build or test)
# exits non-zero.
#
# Examples:
#   tools/mac/swift_test.sh
#   tools/mac/swift_test.sh HandstandCore wt/i38
#   tools/mac/swift_test.sh VisionPose wt/i15
#
# Env: HANDSTAND_MAC_HOST (default macmini).
set -euo pipefail

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "swift_test: $*" >&2; exit 1; }

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

package="${1:-HandstandCore}"
remote_dir="${2:-wt/i38}"
host="${HANDSTAND_MAC_HOST:-macmini}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -d "$repo_root/swift/$package" ]] || die "missing $repo_root/swift/$package"

# The rsync itself lives in swift_sync.sh, shared with real_parity.sh: one
# definition of "copy this worktree's swift/ to the Mac".
source "$repo_root/tools/mac/swift_sync.sh"
swift_sync "$host" "$remote_dir" "$repo_root"

tmp_dir="${TMPDIR:-/tmp/opencode}"
mkdir -p "$tmp_dir"
log="$(mktemp "$tmp_dir/swift_test.XXXXXX")"
trap 'rm -f "$log"' EXIT

echo "== swift build && swift test on $host: $remote_dir/swift/$package"
if ! ssh "$host" "cd '$remote_dir/swift/$package' && swift build && swift test" 2>&1 | tee "$log"; then
    die "swift build/test FAILED on $host in $remote_dir/swift/$package"
fi

# The summary lines the report quotes: XCTest's "Executed … tests" and the
# suites' pass/fail line, plus the Swift Testing equivalent should we ever
# switch frameworks.
echo "== summary"
summary="$(grep -E "Executed [0-9]+ test|Test Suite 'All tests'|Test run with [0-9]+ test" "$log" || true)"
if [[ -z "$summary" ]]; then
    echo "  (no test summary line found in the output above)"
else
    while IFS= read -r line; do echo "  $line"; done <<< "$summary"
fi
