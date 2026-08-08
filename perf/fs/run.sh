#!/usr/bin/env bash
#
# Filesystem overhead under Chimera: native vs. `chimera run` vs. a
# `chimera mount` FUSE view, uncached and cached.
#
# Builds fs_bench and runs each mode four ways against the same file:
# native, under `chimera run` (the userspace VFS behind DBT syscall
# interception), and native through two `chimera mount`s of an empty
# change-set over the live host — one with the default zero-TTL caching
# (every operation revalidated against the live host) and one with
# `--cache 60` (the kernel trusts entries and attributes for a minute).
# Reports ns/op and each route's ratio over native. Point CHIMERA at a
# release build for meaningful numbers (a debug build inflates the
# overhead).
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
ratio() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", a / b }'; }

# Empty change-sets over the live host: each mount serves the same merged
# view a fresh `chimera run` would see, so the routes differ only in how
# the operations reach the VFS and in what the kernel may cache.
FSDIR0="$(mktemp -d)"
FSDIR60="$(mktemp -d)"
MNT0="$(mktemp -d)"
MNT60="$(mktemp -d)"
"$CHIMERA" mount "$FSDIR0" "$MNT0" >/dev/null 2>&1 &
PID0=$!
"$CHIMERA" mount --cache 60 "$FSDIR60" "$MNT60" >/dev/null 2>&1 &
PID60=$!
cleanup() {
    kill -INT "$PID0" "$PID60" 2>/dev/null || true
    wait "$PID0" "$PID60" 2>/dev/null || true
    rm -rf "$FSDIR0" "$FSDIR60" "$MNT0" "$MNT60"
}
trap cleanup EXIT
for _ in $(seq 1 100); do
    mountpoint -q "$MNT0" && mountpoint -q "$MNT60" && break
    sleep 0.05
done
mountpoint -q "$MNT0" && mountpoint -q "$MNT60" ||
    { echo "chimera mount did not come up" >&2; exit 1; }

echo "iters=$ITERS path=$TARGET chimera=$CHIMERA"
printf '%-6s %13s %13s %8s %13s %8s %13s %8s\n' \
    op native chimera ratio fuse ratio fuse-c60 ratio
for op in stat open read; do
    nat=$("$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    chi=$("$CHIMERA" run --rm -- "$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    fus=$("$BUILD/fs_bench" "$op" "$ITERS" "$MNT0$TARGET" | nsop)
    fuc=$("$BUILD/fs_bench" "$op" "$ITERS" "$MNT60$TARGET" | nsop)
    printf '%-6s %10s ns %10s ns %7sx %10s ns %7sx %10s ns %7sx\n' \
        "$op" "$nat" "$chi" "$(ratio "$chi" "$nat")" \
        "$fus" "$(ratio "$fus" "$nat")" "$fuc" "$(ratio "$fuc" "$nat")"
done
