#!/usr/bin/env bash
# The one rsync that carries this worktree's swift/ to the Mac — shared by
# tools/mac/swift_test.sh and tools/mac/real_parity.sh, so "sync the Swift
# package" means the same bytes, the same exclusions and the same rules
# whichever runner is doing it.
#
# Sourced, not executed:
#
#   source "$repo_root/tools/mac/swift_sync.sh"
#   swift_sync "$host" "$remote_dir" "$repo_root"
#
# host        SSH host of the Mac mini (HANDSTAND_MAC_HOST, default macmini)
# remote-dir  directory on the Mac, relative to its home, that receives swift/
# repo_root   this checkout (the caller already knows it)
#
# The whole swift/ directory is copied to $host:$remote-dir/swift/ with
# `rsync -a --delete`, excluding .build and .swiftpm so the remote build cache
# survives; --delete otherwise keeps the remote an exact copy of this
# worktree's swift/. Returns non-zero (and says why) on a bad remote dir or a
# failed rsync.

swift_sync() {
    local host="$1" remote_dir="$2" repo_root="$3"

    [[ -d "$repo_root/swift" ]] || {
        echo "swift_sync: missing $repo_root/swift" >&2
        return 1
    }
    [[ "$remote_dir" != /* && "$remote_dir" != *..* ]] || {
        echo "swift_sync: remote dir must be relative to your home on $host: '$remote_dir'" >&2
        return 1
    }

    echo "== rsync swift/ to $host:$remote_dir/swift/ (excluding .build)"
    rsync -a --delete --exclude '.build' --exclude '.swiftpm' \
        "$repo_root/swift/" "$host:$remote_dir/swift/"
}
