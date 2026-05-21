CARGO_TARGET_DIR := $(shell $(CURDIR)/scripts/cargo-target-dir)
RUNNER ?= $(CARGO_TARGET_DIR)/debug/chimera run

build:
	cargo build --quiet

check: conformance-native conformance
.PHONY: check

conformance: build
	python3 testing/lit.py --runner "$(RUNNER)"
.PHONY: conformance

conformance-native:
	python3 testing/lit.py --runner ""
.PHONY: conformance-native
