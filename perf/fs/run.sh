#!/usr/bin/env bash
#
# Filesystem overhead under Chimera: native vs. `chimera run`.
#
# Builds fs_bench and runs each mode native and under Chimera, reporting ns/op
# and the chimera/native ratio. Point CHIMERA at a release build for meaningful
# numbers (a debug build inflates the overhead).
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

echo "iters=$ITERS path=$TARGET chimera=$CHIMERA"
printf '%-6s %13s %13s %8s\n' op native chimera ratio
for op in stat open read; do
    nat=$("$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    chi=$("$CHIMERA" run -- "$BUILD/fs_bench" "$op" "$ITERS" "$TARGET" | nsop)
    ratio=$(awk -v a="$chi" -v b="$nat" 'BEGIN { printf "%.1f", a / b }')
    printf '%-6s %10s ns %10s ns %7sx\n' "$op" "$nat" "$chi" "$ratio"
done
