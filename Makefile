PY ?= python3
SMOKE_DIR ?= /tmp/xplat_smoke

.PHONY: install test smoke smoke-seq synthetic clean

install:            ## editable install with the test extra
	$(PY) -m pip install -e ".[dev]"

test:               ## unit and regression tests (archive-dependent tests skip when inputs are absent)
	$(PY) -m pytest tests -q

smoke:              ## CPU-only end-to-end run on synthetic data
	$(PY) -m xplat.smoke --work $(SMOKE_DIR)

smoke-seq:          ## also run policy / point-process experiments with a CPU stand-in for mamba_ssm
	$(PY) -m xplat.smoke --work $(SMOKE_DIR) --fake-mamba

synthetic:          ## write synthetic inputs into IO_RESULTS_DIR / TWITTER_TAKEDOWN_DIR
	$(PY) -m xplat.synthetic

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
