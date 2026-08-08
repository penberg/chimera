# The Chimera Manual

Chimera runs unmodified programs in a zero-setup sandbox with a copy-on-write
filesystem. `chimera run` executes the real program, against what looks like
the real filesystem, at close to native speed — but the host does not
change. Every modification the program makes lands in a private *filesystem*:
a change-set you can inspect, discard, build on, or adopt onto the host once
you have seen what it contains.

This manual covers the `chimera` command. The design of the runtime lives in
[ARCHITECTURE.md](ARCHITECTURE.md), and embedding the runtime in your own
program is shown by the worked examples under
[`runtime/examples/`](runtime/examples).

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [How to](#how-to)
- [Reference](#reference)

## Overview

### How it works

Chimera is an ordinary userspace program: no VM, no container, no kernel
module, no privileges. It loads the guest binary into its own address space,
translates its instructions one basic block at a time, and intercepts every
system call the guest makes. Filesystem syscalls are routed through a
userspace overlay mounted at `/`: the guest reads through to the live host
tree, while every write, rename, chmod, and delete lands in the run's private
change-set. The guest sees a normal, complete Linux tree and has no idea it is
sandboxed.

### What the sandbox isolates

`chimera run` confines *filesystem effects*. The guest runs as your user, and
syscalls the overlay does not virtualize — the network among them — are
forwarded to the host kernel. The boundary protects the host's files from the
guest, never the other way around; a guest that must not see a resource at all
needs a policy written against the embedding API, such as the allowlist
sandbox in [`runtime/examples/sandbox/`](runtime/examples/sandbox).

## Quickstart

### Install

```console
$ cargo install --git https://github.com/penberg/chimera chimera-cli
```

From a checkout, use `cargo install --path cli` instead.

### 1. Start a sandboxed shell

Run `chimera run` with no program and you get `/bin/bash`, its prompt badged
with the filesystem the session writes into, so a shell inside a sandbox never
looks like a shell on the host:

```console
$ chimera run
[9f21c07a] ~ $
```

### 2. Change the system

Do something you would never do to a machine you care about:

```console
[9f21c07a] ~ $ echo 'hello from the sandbox' > /etc/hello
[9f21c07a] ~ $ rm /usr/bin/perl
[9f21c07a] ~ $ cat /etc/hello
hello from the sandbox
[9f21c07a] ~ $ perl
bash: perl: command not found
```

Inside the sandbox both changes are real. Exit, and Chimera tells you where
they went:

```console
[9f21c07a] ~ $ exit
chimera: filesystem kept; continue with:
  chimera run --in 9f21c07a
$ cat /etc/hello
cat: /etc/hello: No such file or directory
$ which perl
/usr/bin/perl
```

The host is untouched. The changes live in filesystem `9f21c07a`.

### 3. Inspect the changes

```console
$ chimera fs list
ID         FROM         AGE     SIZE  COMMAND
9f21c07a   host          2m      512B  /bin/bash
$ chimera fs diff 9f21c07a
A /etc/hello
D /usr/bin/perl
```

### 4. Go back in

`--in` resumes the same filesystem — same id, no copy — and the session
picks up exactly where it left off:

```console
$ chimera run --in 9f21c07a
[9f21c07a] ~ $ cat /etc/hello
hello from the sandbox
```

### 5. Adopt or discard

When the changes turn out to be worth keeping, copy them onto the host:

```console
$ chimera fs apply 9f21c07a
A /etc/hello
D /usr/bin/perl
```

When they don't, throw them away:

```console
$ chimera fs rm 9f21c07a
```

That is the whole loop: run, inspect, adopt or discard. The rest of this
manual is detail.

## Concepts

### Filesystems

A *filesystem* is a change-set: everything a run wrote, renamed, chmod'd, or
deleted, recorded separately from the tree it was written against. The guest
experiences it as its entire filesystem, which is where the name comes from —
the change-set is mounted at `/`, layered over whatever it was made against,
so the guest sees one complete tree.

The *host* is simply the filesystem every other one ultimately descends from.
It is the only tree that is live and authoritative, and Chimera never lets a
guest write to it by accident.

### Branch and resume

There are exactly two things you can do with a filesystem, and every run does
one of them:

- **Branch it** (`--from`) — start a *new* filesystem whose changes stack on
  top of the named one. The named filesystem is not modified. This is the
  default, and with no `--from` the thing branched is the host.
- **Resume it** (`--in`) — pick up an existing filesystem where it left off,
  so the run's changes accumulate into it. The named filesystem is modified.

Those two verbs applied to the host and to a kept filesystem cover the whole
surface:

|  | `host` | a filesystem id |
| --- | --- | --- |
| **`--from X`** — branch X; X untouched | new change-set over the live host (the default) | new change-set over that one |
| **`--in X`** — resume X; X modified | mutates the live host — spelled `--unsafe` | resume that change-set |

The bottom-left cell is the only way host state changes without an explicit
adopt step, and because a slip there costs you the machine, Chimera does not
accept `--in host`. That cell has one spelling, `--unsafe`, and `--unsafe`
means exactly `--in host`: no change-set, no sandbox for the filesystem, the
guest writing straight through to the real tree.

The other door onto host state is `chimera fs apply`, which copies a reviewed
change-set onto the host. That is the only merge, and it is the normal way
work escapes a sandbox.

### Nothing is deleted implicitly

Every run keeps its filesystem. Nothing is ever removed behind your back —
not even a change-set that ended up empty, because you may well have branched
precisely in order to have somewhere to stand. `--rm` is the one and only
thing that discards, and you have to ask for it.

### Naming filesystems

How a filesystem is named is independent of what you do with it. A *locator*
names one; the verb — `--from` or `--in` — decides whether it is branched or
resumed. Three forms exist today:

| Locator | Names | Notes |
| --- | --- | --- |
| `host` | The live host tree | Only ever a branch point: `--from host` is the default spelled out, `--in host` is refused — that operation is `--unsafe`. |
| An id | A kept filesystem | 8 hex characters — what `fs list` prints and what the kept notice hands you. The normal currency. |
| A path | A change-set directory, directly | Anything containing `/`. `--in` creates it on first use. Raw state, yours to manage — see [Pin a filesystem in scripts](#pin-a-filesystem-in-scripts). |

`--from` accepts all three, and every `fs` subcommand accepts an id or a
path. `--in` accepts only what can be written to: an id or a path, never
`host`.

Locators of the form `<word>:...` are reserved for filesystems that do not
live on this machine — a container image, a remote change-set. A leading
scheme is recognized before anything else, so adding them later cannot change
how a path or an id is read. Such a source may well be immutable, as a
container image is: it can be branched with `--from` and never resumed with
`--in`. A path whose first component contains a colon needs a leading `./` to
be read as a path.

## How to

### Run a program

`chimera run <program> [args...]` resolves `<program>` through the
filesystem's merged view — a bare name walks `PATH`, so a change-set that
deleted or replaced a binary wins over the host copy — runs it to completion,
and exits with the guest's exit status. Option parsing stops at the program
token, so the guest's own flags need no `--` separator:

```console
$ chimera run ls -la /etc
```

Runs are cheap and independent. Two `chimera run` invocations branch the host
separately and neither can see the other's changes; they are isolated
candidate change-sets over the same live tree.

Every run keeps its filesystem and says so on exit, with the line that takes
you back into it:

```console
$ chimera run touch /etc/hello
chimera: filesystem kept; continue with:
  chimera run --in 51fad6cd
```

When the run changed nothing the notice says so, and the id is still yours to
return to:

```
chimera: filesystem kept (no changes); continue with:
  chimera run --in 51fad6cd
```

For a run you know you do not care about, add `--rm`: it branches the host,
runs, and leaves nothing behind.

```console
$ chimera run --rm make test
```

### Work in a sandboxed shell

Naming no program starts `/bin/bash`. Chimera hands the shell an rc file that
sources your own `~/.bashrc` and then badges the prompt with the filesystem
the session writes into — `unsafe`, on a red field, under `--unsafe` — so the
badge survives a `PS1` set in your dotfiles. Pass `--no-prompt` to leave the
prompt alone.

### Resume a session

`--in` resumes a kept filesystem: same id, no copy, changes accumulating into
it. A session you return to ten times is still one filesystem, not ten.

```console
$ chimera run --in 51fad6cd
```

The `CHIMERA_FS` environment variable supplies a default for `--in`, which is
how you make every command in a shell — or every command an agent issues —
accumulate into one filesystem without threading an id through. Because an
inherited filesystem silently changes both what the guest sees and what it
modifies, a run that takes one from the environment announces it:

```console
$ export CHIMERA_FS=51fad6cd
$ chimera run make install
chimera: --in 51fad6cd (CHIMERA_FS)
```

### Experiment without risk

Use `--from` instead of `--in` when you want to build on a filesystem without
risking it: the run starts a fresh change-set seeded from the named one and
leaves the original exactly as it was, so an experiment that goes wrong costs
you only the branch.

```console
$ chimera run --from 51fad6cd --rm make test
```

That is a throwaway trial on top of kept work: `51fad6cd` cannot be damaged,
and the trial's residue vanishes on exit. Drop `--rm` and the branch is kept
like any other run's filesystem, stacked on `51fad6cd` in `fs list`'s `FROM`
column.

To fork without running anything, `chimera fs branch` prints the new id, so
composition scripts cleanly:

```console
$ chimera run --in $(chimera fs branch 51fad6cd)
```

### Review and adopt changes

`chimera fs diff` shows what a filesystem changed relative to the live host,
one path per line — `A` added, `M` modified, `D` deleted:

```console
$ chimera fs diff 51fad6cd
M /etc/ssh/sshd_config
A /usr/local/bin/tool
```

`chimera fs apply` copies the changes onto the host, and it merges the way a
careful person would. Each modified file remembers the identity of the host
file it was copied from; if the host copy has changed since — an edit, a
deletion, a replacement — the file is a *conflict*: reported, skipped, left
intact on both sides, and the command exits nonzero. Everything conflict-free
applies:

```console
$ chimera fs apply 51fad6cd
A /usr/local/bin/tool
chimera: conflict: /etc/ssh/sshd_config changed on the host since the filesystem change (skipped)
chimera: 1 conflict(s); resolve on the host and re-run apply
```

Resolve on the host and re-run. A re-run recognizes its own earlier work and
never applies a change twice, so `apply` is safe to repeat until it exits
zero.

### Browse a filesystem as a mount

`chimera mount` serves a filesystem's merged view — the live host with the
filesystem's changes applied — at a directory, through FUSE, so ordinary
host tools can browse it without a guest running:

```console
$ mkdir /tmp/view
$ chimera mount 51fad6cd /tmp/view
chimera: mounted 51fad6cd at /tmp/view; unmount with `fusermount -u /tmp/view` or Ctrl-C
```

From another terminal, `/tmp/view/etc/ssh/sshd_config` is the modified copy,
deleted files are absent, and everything the filesystem never touched shines
through from the live host. Writes through the mount land in the change-set,
exactly as a run resuming the filesystem would leave them — the host stays
untouched until `fs apply` — so an editor pointed at the mount edits the
sandbox's state in place. Pass `--read-only` to make the view immutable
instead. The mount counts as a live session: `fs rm` and `fs prune` leave
the filesystem alone until it is unmounted.

### Pin a filesystem in scripts

A path locator names a change-set directory directly, outside the state
directory, which is how you keep one change-set across many `chimera`
invocations without an id at all: point `CHIMERA_FS` at a directory and every
run in that shell accumulates into it. The conformance suite works exactly
this way.

```console
$ export CHIMERA_FS=./build-fs
$ chimera run ./configure
$ chimera run make
```

What a path names is raw state, yours to manage: `--in` creates the directory
on first use, and unlike an id nothing stops you from removing it out from
under a running session.

### Run without a sandbox

`--unsafe` runs with no filesystem at all; the guest mutates the live host,
exactly as if Chimera were not there. It contradicts `--from`, `--in`, and
`--rm`, and it is the only spelling of that operation — `--in host` is
refused and points here. The implicit shell's prompt badge reads `unsafe` so
the one shell that can cost you the machine is the one that looks different.

### Clean up

```console
$ chimera fs rm 51fad6cd
```

removes filesystems by id or path, and is refused while any live session
holds one. To sweep everything no session is using:

```console
$ chimera fs prune
pruning removes these filesystems and their unapplied changes:
  9f21c07a   host          2h      512B  /bin/bash
  51fad6cd   host          3d      12K   make install
remove 2 filesystem(s)? [y/N] y
removed 2 filesystem(s), freed 12K
```

`prune` lists the candidates and asks first, because removal takes their
unapplied changes with them; `-f` skips the prompt.

## Reference

### `chimera run`

Run a program in a sandbox.

```
chimera run [OPTIONS] [PROGRAM [ARG...]]
```

With no `PROGRAM`, runs `/bin/bash` with a badged prompt. Option parsing
stops at the program token, so everything after it belongs to the guest.
Options are spelled `--name value`, not `--name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `--from`, `-f` `<filesystem>` | `host` | Branch point: `host`, a kept filesystem's id, or a path to a change-set directory. The named filesystem is left exactly as it was. |
| `--in <filesystem>` | | Resume an existing filesystem: an id or a path, never `host`. Changes accumulate into it rather than into a new one. Defaults from `CHIMERA_FS`. |
| `--rm` | | Discard the new filesystem when the run exits. Only meaningful when branching; refused with `--in`, where it would destroy a filesystem you had deliberately kept. |
| `--unsafe` | | No filesystem at all; the guest mutates the live host. Contradicts `--from`, `--in`, and `--rm`. |
| `--no-prompt` | | Leave the started shell's prompt alone instead of badging it. |
| `--code-cache-size <MiB>` | 256 | Capacity of the translated-code cache. |

`run` exits with the guest's exit status. On exit, a run that branched prints
the kept notice naming its filesystem; a run given `--in` from the
environment announces `chimera: --in <fs> (CHIMERA_FS)` at startup. The badge
falls back to a plain `[id]` prefix when `TERM` is `dumb` or `NO_COLOR` is
set.

### `chimera fs`

Tooling over kept filesystems. It reads the self-describing on-disk format
directly; there is no daemon and no index. Every subcommand accepts an id or
a path.

#### `chimera fs list`

Every kept filesystem, newest first: id, the filesystem it branched from
(`host` for a run that branched the live host), age, change-set size, and the
command that created it.

```console
$ chimera fs list
ID         FROM         AGE     SIZE  COMMAND
51fad6cd   host          3d      12K   make install
80ffe11b   51fad6cd      2h      1K    /bin/bash
```

#### `chimera fs diff <filesystem>`

What the filesystem changed relative to the live host, one path per line:
`A` added, `M` modified, `D` deleted.

#### `chimera fs apply <filesystem>`

Copy the changes onto the host — the adopt step. Prints each applied change
in `diff`'s format. A modified file whose host copy no longer matches the
identity recorded when the filesystem copied it up is a conflict: reported on
stderr, skipped, and left intact on both sides, and the command exits
nonzero. A re-run recognizes changes it has already applied and refuses,
rather than repeats, anything the host has since overruled.

#### `chimera fs branch <src>`

Fork a filesystem without running anything; prints the new id on stdout, so
`chimera run --in $(chimera fs branch 51fad6cd)` scripts cleanly. The copy is
exact — a branch diffs and applies precisely as its source would have — and
because it is a copy rather than a reference, later resuming the source with
`--in` cannot disturb anything already branched from it. Branch from a
quiesced source: a live session still writing into it can tear the snapshot.

#### `chimera fs rm <filesystem>...`

Remove filesystems. Refused while any live session holds one.

#### `chimera fs prune`

Remove every filesystem no live session is using, after listing the
candidates and confirming. `-f` skips the prompt.

### `chimera mount`

Mount a filesystem's merged view at a directory, through FUSE.

```
chimera mount [--read-only] <filesystem> <mountpoint>
```

The view is what a run resuming the filesystem would see: the live host
below, the change-set above. Writes through the mount land in the
change-set, so `fs diff` reports them and `fs apply` adopts them like any
other change; `--read-only` refuses writes at the mount. The filesystem is
named as `--in` names one — an id names a kept filesystem, a path names a
change-set directory, created on first use — and the mountpoint must be an
existing directory.

The command runs in the foreground until the filesystem is unmounted, by
`fusermount -u`, `umount`, or Ctrl-C. While mounted it counts as a live
session, so `fs rm` and `fs prune` refuse to remove the filesystem. Nothing
is cached: the kernel revalidates every entry and attribute against the
merged view, so changes on the live host appear in the mount as they happen.
One caveat of the path-keyed view: two hard links to one file report two
inode numbers (their link counts still agree), so tools that deduplicate by
inode see them as distinct files.

### `chimera version`

Print the version, target, and whether memory-protection keys back the code
cache (`mpk` or `nompk`).

### Environment

| Variable | Effect |
| --- | --- |
| `CHIMERA_FS` | Default for `--in`. A run that takes it announces `chimera: --in <fs> (CHIMERA_FS)` on stderr. Inert under `--unsafe`. |
| `XDG_STATE_HOME` | Overrides where kept filesystems live (see [Files](#files)). |

### Exit status

`chimera run` exits with the guest's exit status. Chimera's own failures exit
1 with a message on stderr prefixed `chimera:`. `chimera fs apply` exits
nonzero when any change conflicted.

### Files

Kept filesystems live under `$XDG_STATE_HOME/chimera/fs` (defaulting to
`~/.local/state`), one directory per id:

| Entry | Contents |
| --- | --- |
| `data/` | The change-set itself, in the format the runtime owns. |
| `tmp/` | Staging space. |
| `meta` | Human-readable provenance — informational only, including the `parent` line a branch records. |
| `lock` | Coordinates live sessions and removal. |

The change-set encodes deletions and metadata in user extended attributes, so
the state directory must sit on a filesystem with user-xattr support.
Branching copies the change-set, reflinked where the underlying filesystem
supports it, as Btrfs and XFS do.
