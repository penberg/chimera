#!/usr/bin/env bash
#
# Filesystem overhead under Chimera: native vs. `chimera run` vs. a
# `chimera mount` FUSE view.
#
# Builds fs_bench and runs each mode three ways against the same file:
# native, under `chimera run` (the userspace VFS behind DBT syscall
# interception), and native through a `chimera mount` of an empty change-set
# over the live host (the same merged view, reached through the kernel's
# FUSE protocol instead). Reports ns/op and each route's ratio over native.
# Point CHIMERA at a release build for meaningful numbers (a debug build
# inflates the overhead).
#
#   ./run.sh
#   CHIMERA=../../target/release/chimera ./run.sh [iters] [path]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$DIR/build"
CHIMERA="${CHIMERA:-chimera}"
ITERS="${1:-200000}"
TARGET="${2:-/usr/lib/os-release}"

mkdir -p "$BUILD"
cc -O2 -o "$BUILD/fs_bench" "$DIR/fs_bench.c"
nsop() { awk '{print $2}'; }

# An empty change-set over the live host: the mount serves the same merged
# view a fresh `chimera run` would see, so the two routes differ only in how
# the operations reach the VFS.
FSDIR="$(mktemp -d)"
MNT="$(mktemp -d)"
"$CHIMERA" mount "$FSDIR" "$MNT" >/dev/null 2>&1 &
MOUNT_PID=$!
cleanup() {
    kill -INT "$MOUNT_PID" 2>/dev/null || true
    wait "$MOUNT_PID" 2>/dev/null || true
    rm -rf "$FSDIR" "$MNT"
}
trap cleanup EXIT
for _ in $(seq 1 100); do
    mountpoint -q "$MNT" && break
    sleep 0.05
done
mountpoint -q "$MNT" || { echo "chimera mount did not come up" >&2; exit 1; }

echo "iters=$ITERS path=$TARGET chimera=$CHIMERA"
printf '%-6s %13s %13s %8s %13s %8s\n' op native chimera ratio fuse ratio
for op in stat open read; do
    nat=$("$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    chi=$("$CHIMERA" run --rm -- "$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    fus=$("$BUILD/fs_bench" "$op" "$ITERS" "$MNT$TARGET" | nsop)
    cratio=$(awk -v a="$chi" -v b="$nat" 'BEGIN { printf "%.1f", a / b }')
    fratio=$(awk -v a="$fus" -v b="$nat" 'BEGIN { printf "%.1f", a / b }')
    printf '%-6s %10s ns %10s ns %7sx %10s ns %7sx\n' \
        "$op" "$nat" "$chi" "$cratio" "$fus" "$fratio"
done
