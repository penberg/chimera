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

# The suite through a release build of the runtime. Optimized code is what a
# user actually runs, and it is not interchangeable with the debug build for
# testing: the one known intermittent failure reproduces only here, so a green
# `conformance` is not evidence about it.
conformance-release:
	cargo build --release --quiet
	python3 testing/lit.py --runner "$(CARGO_TARGET_DIR)/release/chimera run"
.PHONY: conformance-release

# The suite with the copy-on-write filesystem bypassed: the guest mutates the
# host directly, so this pins the passthrough path the overlay normally shields.
conformance-unsafe: build
	python3 testing/lit.py --runner "$(RUNNER) --unsafe"
.PHONY: conformance-unsafe
