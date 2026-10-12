CARGO_TARGET_DIR := $(shell $(CURDIR)/scripts/cargo-target-dir)
RUNNER ?= $(CARGO_TARGET_DIR)/debug/chimera run

# Nothing about the copy-on-write filesystem is Linux-only; it has simply not
# been built for anything else yet. So its tests stay unconditioned in the tree
# and are dropped here instead, until the Darwin port catches up.
EXCLUDE := $(if $(filter Darwin,$(shell uname -s)),--exclude fs,)

# The backends `make conformance` runs the suite under: the translator, and
# syscall user dispatch where the host can run it (x86-64 Linux 5.11 or
# later), which the freshly built Chimera finds out by running a program
# under it. Run one with `make conformance BACKENDS=sud`.
BACKENDS = dbt $(shell $(RUNNER) --backend sud --unsafe /bin/true >/dev/null 2>&1 && echo sud)

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
	@for backend in $(BACKENDS); do \
	    echo "# backend: $$backend"; \
	    CHIMERA_BACKEND=$$backend python3 testing/lit.py $(EXCLUDE) --runner "$(RUNNER)" || exit 1; \
	done
.PHONY: conformance

conformance-native:
	python3 testing/lit.py $(EXCLUDE) --runner ""
.PHONY: conformance-native

# The suite with the copy-on-write filesystem bypassed: the guest mutates the
# host directly, so this pins the passthrough path the overlay normally shields.
conformance-unsafe: build
	python3 testing/lit.py $(EXCLUDE) --runner "$(RUNNER) --unsafe"
.PHONY: conformance-unsafe
