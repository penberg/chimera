CARGO_TARGET_DIR := $(shell $(CURDIR)/scripts/cargo-target-dir)
RUNNER ?= $(CARGO_TARGET_DIR)/debug/chimera run

# Nothing about the copy-on-write filesystem is Linux-only; it has simply not
# been built for anything else yet. So its tests stay unconditioned in the tree
# and are dropped here instead, until the Darwin port catches up.
EXCLUDE := $(if $(filter Darwin,$(shell uname -s)),--exclude fs,)

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
	python3 testing/lit.py $(EXCLUDE) --runner "$(RUNNER)"
.PHONY: conformance

conformance-native:
	python3 testing/lit.py $(EXCLUDE) --runner ""
.PHONY: conformance-native

# The suite with the copy-on-write filesystem bypassed: the guest mutates the
# host directly, so this pins the passthrough path the overlay normally shields.
conformance-unsafe: build
	python3 testing/lit.py $(EXCLUDE) --runner "$(RUNNER) --unsafe"
.PHONY: conformance-unsafe

# The suite against the syscall-user-dispatch backend, which runs the guest
# natively. Not part of `check`: the backend is a proof of concept and the
# thread and signal groups are outside its scope, so a full pass is not yet
# the expectation.
conformance-sud: build
	python3 testing/lit.py --runner "$(RUNNER) --backend sud"
.PHONY: conformance-sud
