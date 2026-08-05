# Copy-on-write filesystem tests

The overlay that gives every run its own delta: copy-up, whiteouts, origin
records, the `chimera fs` tooling that inspects and replays them.

The directory carries no condition: nothing about copy-on-write is
Linux-specific. The implementation merely is — the whiteout and origin markers
are xattrs, and the VFS underneath leans on `openat2`, `O_PATH` and
`renameat2` — so the Makefile excludes these tests on Darwin until the port
catches up. That exclusion is the thing to delete when it does.
