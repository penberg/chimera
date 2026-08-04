CARGO_TARGET_DIR := $(shell $(CURDIR)/scripts/cargo-target-dir)
RUNNER ?= $(CARGO_TARGET_DIR)/debug/chimera run

build:
	cargo build --quiet

check: conformance-native conformance
.PHONY: check

clean:
	cargo clean
.PHONY: clean

# Run the GitHub Actions CI workflow locally with Agent CI (https://agent-ci.dev/).
ci:
	npx -y @redwoodjs/agent-ci run --workflow .github/workflows/ci.yml
.PHONY: ci

conformance: build
	python3 testing/lit.py --runner "$(RUNNER)"
.PHONY: conformance

conformance-native:
	python3 testing/lit.py --runner ""
.PHONY: conformance-native

# Darwin/arm64: run the host-portable conformance subset through chimera. The
# ABI tests carry aarch64/apple blocks; the signals, threads, and exec suites
# run their portable subsets, with the Linux-specific tests (clone,
# real-time/sigtimedwait, execveat, x86-asm) carrying `UNSUPPORTED: darwin` so
# lit skips rather than fails them. The linux/ suite remains Linux-specific and
# is excluded here. Default cc on Apple Silicon already targets arm64.
conformance-darwin: build
	python3 testing/lit.py --runner "$(RUNNER)" \
		testing/conformance/abi testing/conformance/exec \
		testing/conformance/signals testing/conformance/threads
.PHONY: conformance-darwin

# The suite with the copy-on-write filesystem bypassed: the guest mutates the
# host directly, so this pins the passthrough path the overlay normally shields.
conformance-unsafe: build
	python3 testing/lit.py --runner "$(RUNNER) --unsafe"
.PHONY: conformance-unsafe
