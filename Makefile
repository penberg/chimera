RUNNER ?= $(CURDIR)/target/debug/chimera run

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
