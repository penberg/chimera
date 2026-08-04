#!/usr/bin/env python3
"""Minimal LIT-style runner for chimera conformance tests.

Each test source file embeds one or more directives of the form:

    // RUN: <shell command>

The runner walks `testing/conformance/`, parses RUN lines from each `.c`
file, expands a small set of substitutions, and runs each command under
`sh -c`. A test passes when every RUN line exits 0.

Substitutions:
    %s        path to the test source file
    %t        path to a per-test scratch file (no extension)
    %cc       C compiler           (env: $CC,     default: `cc`)
    %runner   command prefix used to invoke a built test binary
              (env: $RUNNER, default: empty)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = REPO_ROOT / "testing" / "conformance"

RUN_RE = re.compile(r"//\s*RUN:\s*(.+?)\s*$")
REQUIRES_RE = re.compile(r"//\s*REQUIRES:\s*(.+?)\s*$")
UNSUPPORTED_RE = re.compile(r"//\s*UNSUPPORTED:\s*(.+?)\s*$")
XFAIL_RE = re.compile(r"//\s*XFAIL:\s*(.+?)\s*$")

_USE_COLOR = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
GREEN = lambda s: _c("32", s)
RED   = lambda s: _c("31", s)
DIM   = lambda s: _c("2",  s)


def parse_runs(path: Path) -> list[str]:
    runs: list[str] = []
    for line in path.read_text().splitlines():
        m = RUN_RE.search(line)
        if m:
            runs.append(m.group(1).strip())
    return runs


def _feature_list(match: re.Match) -> list[str]:
    # Everything after `--` is a human-readable reason, not a feature.
    body = match.group(1).split("--", 1)[0]
    return [f.strip() for f in body.replace(",", " ").split() if f.strip()]


def gating(path: Path) -> tuple[list[str], list[str], list[str]]:
    """The test's `REQUIRES:`, `UNSUPPORTED:`, and `XFAIL:` feature lists
    (unioned across lines). The first two decide whether it runs under the
    active feature set; `XFAIL` marks features under which it is expected to
    fail (a known, tracked bug)."""
    requires: list[str] = []
    unsupported: list[str] = []
    xfail: list[str] = []
    for line in path.read_text().splitlines():
        if m := REQUIRES_RE.search(line):
            requires += _feature_list(m)
        if m := UNSUPPORTED_RE.search(line):
            unsupported += _feature_list(m)
        if m := XFAIL_RE.search(line):
            xfail += _feature_list(m)
    return requires, unsupported, xfail


def substitute(cmd: str, *, source: Path, tmp: Path, cc: str, runner: str) -> str:
    return (cmd
        .replace("%cc", cc)
        .replace("%runner", runner)
        .replace("%s", str(source))
        .replace("%t", str(tmp)))


@dataclass
class Result:
    status: str   # "pass" | "fail" | "skip" | "xfail" | "xpass"
    detail: str = ""


def run_test(source: Path, *, cc: str, runner: str, timeout: float, features: set[str]) -> Result:
    requires, unsupported, xfail = gating(source)
    if missing := [f for f in requires if f not in features]:
        return Result("skip", f"requires {', '.join(missing)}")
    if blocked := [f for f in unsupported if f in features]:
        return Result("skip", f"unsupported on {', '.join(blocked)}")
    expect_fail = [f for f in xfail if f in features]
    runs = parse_runs(source)
    if not runs:
        return Result("skip", "no RUN directives")
    failure: str | None = None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / source.stem
        # Fresh filesystems land under $XDG_STATE_HOME/chimera/fs;
        # point that at the per-test directory so every run's filesystem is
        # born and dies with the test.
        env = dict(os.environ)
        env["XDG_STATE_HOME"] = str(Path(td) / "state")
        for cmd in runs:
            full = substitute(cmd, source=source, tmp=tmp, cc=cc, runner=runner)
            try:
                proc = subprocess.run(
                    ["sh", "-c", full],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                # A test that never returns (e.g. a deadlock) is a failure, not
                # an excuse to hang the whole suite forever.
                failure = f"timed out after {timeout:g}s\n  cmd: {full}"
                break
            if proc.returncode != 0:
                detail = f"exit={proc.returncode}\n  cmd: {full}\n"
                if proc.stdout.strip():
                    detail += f"  stdout:\n{_indent(proc.stdout, '    ')}\n"
                if proc.stderr.strip():
                    detail += f"  stderr:\n{_indent(proc.stderr, '    ')}"
                failure = detail.rstrip()
                break

    # A test marked `XFAIL` for an active feature is expected to fail: a failure
    # is the tracked outcome (`xfail`), and a pass is an `xpass` — the bug got
    # fixed, so the marker is now stale and the run is flagged so it gets removed.
    if expect_fail:
        if failure is None:
            return Result("xpass", f"unexpectedly passed (drop the XFAIL: {', '.join(expect_fail)})")
        return Result("xfail", f"expected failure on {', '.join(expect_fail)}")
    return Result("fail", failure) if failure is not None else Result("pass")


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.rstrip("\n").splitlines())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cc",
        default=os.environ.get("CC", "cc"),
        help="C compiler to substitute for %%cc (env: CC, default: cc)",
    )
    p.add_argument(
        "--runner",
        default=os.environ.get("RUNNER", ""),
        help="command prefix to substitute for %%runner (env: RUNNER, default: empty)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("LIT_TIMEOUT", "120")),
        help="per-RUN-line timeout in seconds; a slower run fails (env: LIT_TIMEOUT, default: 120)",
    )
    p.add_argument(
        "--feature",
        action="append",
        default=[],
        metavar="NAME",
        help="declare an available feature for REQUIRES:/UNSUPPORTED: gating "
        "(repeatable); the host platform — 'darwin' or 'linux' — is always included",
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="restrict to these test files or directories (default: all of testing/conformance)",
    )
    args = p.parse_args()

    # The host platform is always an implicit feature, so a test can carry
    # `UNSUPPORTED: darwin` (or `REQUIRES: linux`) with no extra flags. Running
    # under a runner adds `chimera` and `<platform>-chimera`, because "expected
    # to fail" is often a property of the translator on one host rather than of
    # the host itself: a native run of the same test has no translator and would
    # report a stale `XFAIL` as an `XPASS`.
    platform = "darwin" if sys.platform == "darwin" else "linux"
    features = {platform, *args.feature}
    if args.runner:
        features |= {"chimera", f"{platform}-chimera"}

    roots = args.paths if args.paths else [TEST_ROOT]
    tests: list[Path] = []
    for r in roots:
        r = r.resolve()
        if r.is_file():
            tests.append(r)
        elif r.is_dir():
            tests.extend(sorted(r.rglob("*.c")))
        else:
            print(f"warning: {r} does not exist", file=sys.stderr)

    if not tests:
        print("no tests found", file=sys.stderr)
        return 1

    print(
        DIM(f"# cc={args.cc} runner={args.runner or '<none>'} features={','.join(sorted(features))}"),
        file=sys.stderr,
    )

    passed = failed = skipped = xfailed = xpassed = 0
    for t in tests:
        rel = t.relative_to(REPO_ROOT)
        res = run_test(t, cc=args.cc, runner=args.runner, timeout=args.timeout, features=features)
        if res.status == "pass":
            passed += 1
            print(f"{GREEN('PASS')}  {rel}")
        elif res.status == "skip":
            skipped += 1
            print(f"{DIM('SKIP')}  {rel}  ({res.detail})")
        elif res.status == "xfail":
            xfailed += 1
            print(f"{DIM('XFAIL')} {rel}  ({res.detail})")
        elif res.status == "xpass":
            # An unexpected pass is a failure of the suite: the marker is stale.
            xpassed += 1
            print(f"{RED('XPASS')} {rel}  ({res.detail})")
        else:
            failed += 1
            print(f"{RED('FAIL')}  {rel}")
            if res.detail:
                print(_indent(res.detail, "    "))

    total = passed + failed + skipped + xfailed + xpassed
    parts = [f"{passed} passed", f"{failed} failed", f"{skipped} skipped"]
    if xfailed:
        parts.append(f"{xfailed} xfailed")
    if xpassed:
        parts.append(f"{xpassed} xpassed")
    parts.append(f"{total} total")
    summary = ", ".join(parts)
    bad = failed + xpassed
    print()
    print(summary if bad == 0 else RED(summary))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
