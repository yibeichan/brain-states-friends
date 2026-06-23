"""Audit + diagnostic tests for the ICA convergent-validity supplement.

Data-free tests run on the branch alone (synthetic inputs). The data-gated
determinism guard skips when SCRATCH outputs are absent.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # put script/ on path


def test_nojax_unpickler_substitutes_jax_class_only():
    from utils.jax_free_model_io import _NoJaxUnpickler, _HMMStub
    up = _NoJaxUnpickler(io.BytesIO(b""))
    # any module path containing 'hdphmm_jax' -> stub
    assert up.find_class("project.utils.hdphmm_jax", "StickyHDPHMM_JAX") is _HMMStub
    # unrelated classes resolve normally
    assert up.find_class("numpy", "ndarray") is np.ndarray
