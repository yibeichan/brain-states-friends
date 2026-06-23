"""JAX-free loader for pickled StickyHDPHMM_JAX models.

# SUPPLEMENT-ONLY: absent from main; do not sync. Lets the supplement load the
# fitted HMM (means_/covars_/transmat_/startprob_) without importing jax, which
# is not part of the supplement's uv environment.
"""
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class _HMMStub:
    """Minimal stub holding the numpy arrays from a pickled StickyHDPHMM_JAX."""
    pass


class _NoJaxUnpickler(pickle.Unpickler):
    """Substitutes any 'hdphmm_jax' class with _HMMStub so jax is never imported."""
    def find_class(self, module, name):
        if "hdphmm_jax" in module:
            return _HMMStub
        return super().find_class(module, name)


def _load_model_no_jax(path):
    """Load a pickled StickyHDPHMM (or _JAX) without requiring jax."""
    with open(path, "rb") as f:
        model = _NoJaxUnpickler(f).load()
    if isinstance(model, _HMMStub):
        for attr in ("means_", "covars_", "transmat_", "startprob_",
                     "n_components", "covariance_type"):
            if not hasattr(model, attr):
                raise RuntimeError(
                    f"_HMMStub missing attribute '{attr}' after load: "
                    f"check that the model pickle contains this field.")
        for attr in ("means_", "covars_", "transmat_", "startprob_"):
            val = getattr(model, attr)
            if val is not None and not isinstance(val, np.ndarray):
                setattr(model, attr, np.asarray(val))
        cov = np.asarray(model.covars_)
        if cov.ndim not in (2, 3):
            raise ValueError(
                f"loaded model covars_ has unexpected ndim={cov.ndim} "
                f"(expected 2=diag or 3=full); _normalize_covars was skipped")
        logger.info("Loaded JAX model via _HMMStub (no jax import needed); "
                    "n_components=%d, covariance_type=%s",
                    model.n_components, model.covariance_type)
    return model
