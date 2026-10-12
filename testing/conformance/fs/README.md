# Copy-on-write filesystem tests

The overlay that gives every run its own delta: copy-up, whiteouts, origin
records, the `chimera fs` tooling that inspects and replays them.

The directory carries no condition: the overlay runs on every host, with the
host's own spelling of what the format needs — extended attributes, an
exclusive and a swapping rename, a confined directory walk — behind one small
seam per host in the runtime. A test here that reaches for an interface by
name (`getxattr`, `renameat2`, `/proc/self/exe`) carries the Darwin spelling
beside the Linux one.
