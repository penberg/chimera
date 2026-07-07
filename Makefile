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

# The whole suite under the copy-on-write overlay bring-up toggle: each test
# gets a fresh delta, and the overlay over nothing must be indistinguishable
# from the host (OVERLAYFS.md task 3).
conformance-cow: build
	python3 testing/lit.py --runner "$(RUNNER)" --cow
.PHONY: conformance-cow
