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
#
# Each cell is measured over ROUNDS self-timed rounds of ITERS operations in
# one process (one warm-up, many samples), reported as mean±sd ns/op with the
# ratio of means over native. Every raw per-round sample also lands in
# build/samples.csv as `route,op,round,ns_per_op`, ready for error bars.
# Point CHIMERA at a release build for meaningful numbers (a debug build
# inflates the overhead).
#
#   ./run.sh
#   CHIMERA=../../target/release/chimera ./run.sh [iters-per-round] [path] [rounds]
set -euo pipefail
export LC_ALL=C

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$DIR/build"
CHIMERA="${CHIMERA:-chimera}"
ITERS="${1:-20000}"
TARGET="${2:-/usr/lib/os-release}"
ROUNDS="${3:-10}"
CSV="$BUILD/samples.csv"

mkdir -p "$BUILD"
cc -O2 -o "$BUILD/fs_bench" "$DIR/fs_bench.c"
echo "route,op,round,ns_per_op" > "$CSV"

# Record one cell: run the given command (one process, ROUNDS timed rounds),
# append each round's ns/op to the CSV, and echo "mean sd" of the samples.
sample() {
    local route=$1 op=$2
    shift 2
    local vals
    vals=$("$@" | awk '{print $2}')
    local i=0 v
    while read -r v; do
        i=$((i + 1))
        echo "$route,$op,$i,$v" >> "$CSV"
    done <<< "$vals"
    echo "$vals" | awk '
        { s += $1; ss += $1 * $1; n++ }
        END {
            m = s / n
            sd = (n > 1) ? sqrt((ss - n * m * m) / (n - 1)) : 0
            printf "%.1f %.1f\n", m, sd
        }'
}

cell() { printf '%.0f±%.0f' "$1" "$2"; }
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

echo "iters=$ITERS/round rounds=$ROUNDS path=$TARGET chimera=$CHIMERA"
echo "ns/op mean±sd; ratios of means over native; samples in ${CSV#"$DIR"/}"
printf '%-6s %15s %15s %6s %15s %6s %15s %6s\n' \
    op native chimera ratio fuse ratio fuse-c60 ratio
for op in stat open read; do
    read -r natm natsd < <(sample native "$op" \
        "$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" "$ROUNDS")
    read -r chim chisd < <(sample chimera "$op" \
        "$CHIMERA" run --rm -- "$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" "$ROUNDS")
    read -r fusm fussd < <(sample fuse "$op" \
        "$BUILD/fs_bench" "$op" "$ITERS" "$MNT0$TARGET" "$ROUNDS")
    read -r fucm fucsd < <(sample fuse-c60 "$op" \
        "$BUILD/fs_bench" "$op" "$ITERS" "$MNT60$TARGET" "$ROUNDS")
    printf '%-6s %15s %15s %5sx %15s %5sx %15s %5sx\n' \
        "$op" \
        "$(cell "$natm" "$natsd")" \
        "$(cell "$chim" "$chisd")" "$(ratio "$chim" "$natm")" \
        "$(cell "$fusm" "$fussd")" "$(ratio "$fusm" "$natm")" \
        "$(cell "$fucm" "$fucsd")" "$(ratio "$fucm" "$natm")"
done
