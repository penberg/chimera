# Runner image for `make ci` (Agent CI, https://agent-ci.dev/).
#
# Agent CI's default runner image, ghcr.io/actions/actions-runner:latest, is a
# minimal container that ships neither system build tools nor a Rust toolchain
# (unlike GitHub's hosted ubuntu-latest VM). This image layers in what the CI
# workflow needs: a C compiler and make for the conformance suite, python3 to
# drive testing/lit.py, and rustup with the pinned toolchain.
#
# Agent CI hashes this file and caches the built image, so edits rebuild it
# (~60-90s) and unchanged runs reuse the cache.
#
# Known local-only failures: Docker's default seccomp profile answers `clone3`
# with ENOSYS, so `testing/conformance/threads/clone3-bad-args.c` and
# `clone3-clear-sighand.c` fail under `make ci` in every mode — including the
# native one, which runs no Chimera at all. GitHub's hosted runners are VMs
# rather than containers and have the syscall, so these pass in real CI. To
# confirm a failure is the sandbox and not the code, run the suite directly:
#
#   docker run --rm --security-opt seccomp=unconfined -v "$PWD":/w -w /w \
#     <image> python3 testing/lit.py --runner ""
#
# Agent CI exposes no way to relax the profile for its own runs.
FROM ghcr.io/actions/actions-runner:latest

RUN sudo apt-get update \
 && sudo apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      python3 \
      curl \
      ca-certificates \
 && sudo rm -rf /var/lib/apt/lists/*

# Install rustup and the toolchain. Keep the channel and components in sync with
# rust-toolchain.toml; rustup also honours that file at runtime, but installing
# here bakes the toolchain into the cached image instead of re-fetching per run.
#
# The GitHub Actions runner does not propagate this image's ENV PATH to `run:`
# step shells, so symlink the rustup proxies into /usr/local/bin, which is on
# the runner's default PATH. The proxies dispatch on argv[0], so a cargo ->
# rustup symlink still resolves to the cargo toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.95.0 \
                 --component rustfmt,clippy \
 && sudo ln -s /home/runner/.cargo/bin/* /usr/local/bin/
