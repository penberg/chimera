#!/usr/bin/env bash
#
# Steady-state DBT overhead: native vs. under Chimera.
#
# Builds each perf-<name>.c microbenchmark and runs it both natively and under
# Chimera, reporting the self-timed ns/op (startup and one-time translation
# excluded; see perf.h) and the chimera/native ratio. Each workload isolates
# one guest execution pattern the translator must handle. Set CHIMERA to point
# at the binary (default: PATH).
#
#   ./run.sh
#   CHIMERA=../target/release/chimera ./run.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$DIR/build"
CHIMERA="${CHIMERA:-chimera}"

# Logical order: in-cache edges first, then boundary-crossing, then the control.
WORKLOADS=(loop direct indirect indirect-mega ret callsites syscall signal mem)

mkdir -p "$BUILD"
nsop() { awk '{print $2}'; }

printf '%-14s %13s %13s %8s\n' workload native chimera ratio
for w in "${WORKLOADS[@]}"; do
    bin="$BUILD/$w"
    cc -O2 -I "$DIR/.." -o "$bin" "$DIR/perf-$w.c"
    nat=$("$bin" | nsop)
    chi=$("$CHIMERA" run -- "$bin" | nsop)
    ratio=$(awk -v a="$chi" -v b="$nat" 'BEGIN { printf "%.1f", a / b }')
    printf '%-14s %10s ns %10s ns %7sx\n' "$w" "$nat" "$chi" "$ratio"
done
