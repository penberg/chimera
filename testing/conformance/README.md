# Conformance tests

Each directory here is named `<topic>[-<condition>...]`. The topic says what
the tests are about; the conditions, if any, say where they run. Nothing
inside a test file decides that — a test carries no marker naming a host it
does not like, and the runner never reports a test as skipped for being on
the wrong one. It is simply not collected.

The conditions are `linux`, `darwin`, `x86` and `arm64`, and they compose:

    signals/               every host
    signals-linux/         Linux only
    signals-x86/           x86-64 only, either OS
    isolation-linux-x86/   x86-64 Linux only

Chimera translates same-ISA, so the guest's architecture is always the host's
and one pair of conditions covers both.

Topic names carry no hyphen, so every segment after the first must name a
condition. `signals-x86_64/` is an error the runner reports rather than a
directory that quietly runs everywhere.

## Where a new test goes

Put it in the bare topic directory unless it *cannot* run elsewhere, and let
the reason be visible in the source: an inline-asm block with x86 register
constraints, a Linux-only syscall, `/proc`. A test that merely fails on
another host is a bug to fix or an unported feature — not a condition.

## Expected failures

`// XFAIL: <condition>…` is the one marker a test still carries, and it says
something a directory cannot: the test *should* pass here and does not. That is
a tracked bug, not a boundary.

Beyond the host's own conditions, a run under Chimera also offers `chimera` and
`<host>-chimera`, because a defect is usually the translator's rather than the
host's — `XFAIL: darwin-chimera` expects the failure under `make conformance` on
macOS while still demanding a pass from `make conformance-native`. When the bug
is fixed the run reports an `XPASS` and the marker comes out.

`XFAIL` says *deterministically* fails, so it is the wrong tool for a flake: a
test that fails one run in ten would report `XPASS` — and fail the suite — the
other nine times. An intermittent failure has to be fixed or left failing where
it is visible; there is no marker that makes it quiet and honest at once.

## The filesystem

`fs/` is the standing example. The copy-on-write filesystem exists only on
Linux today, but nothing about the design is Linux-specific, so its tests are
unconditioned and the Makefile drops them on Darwin (`--exclude fs`) until the
port catches up. Encoding that as `fs-linux/` would have claimed something
untrue about the feature.
