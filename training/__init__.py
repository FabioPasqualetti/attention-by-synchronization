"""Shared support library: training backends, data loaders, run harness, path config.

Every experiment driver imports from here; there is one training backend, not per-experiment
copies. ``training/selftest.py`` checks that the derived modules match the stock
``oscillator_attention`` ones exactly."""
from . import paths  # noqa: F401
paths.ensure_paths()
