"""Pure math for the ICA out-of-stimulus recurrence supplement
(`sm_alt_ica_oos_recurrence`). No I/O.

The ICA analogue of Results R5: a Friends-fit ICA component's recurrence rank
vs its occupancy on out-of-stimulus data, with the ICA applied frozen (no refit).
WTA labelling (per-run z-score then signed argmax) makes ICA components the
discrete-state analogue, so the HMM recurrence definition (05a) transfers
unchanged.
"""
import numpy as np

from .ica_states import wta_labels  # re-exported for callers' convenience

__all__ = ["wta_labels", "fo_per_run", "recurrence_scores", "continuous_occupancy"]


def fo_per_run(labels, run_boundaries, n_components):
    """Run-level fractional occupancy of WTA labels.

    Returns {run_index: np.ndarray(n_components,)} where entry k is the fraction
    of that run's TRs whose WTA winner is component k.
    """
    labels = np.asarray(labels)
    fo = {}
    for i, (s0, e0) in enumerate(run_boundaries):
        seg = labels[s0:e0]
        counts = np.bincount(seg, minlength=n_components).astype(float)
        fo[i] = counts / len(seg) if len(seg) else np.zeros(n_components)
    return fo


def recurrence_scores(fo, n_components, fo_threshold):
    """Fraction of runs in which each component is 'active' (FO > threshold).

    Vendored from 05a recurrence_utils.compute_recurrence_scores (that module is
    not present on the supplements branch). recurrence_k in [0, 1].
    """
    n_units = len(fo)
    if n_units == 0:
        return np.zeros(n_components)
    active = np.zeros(n_components)
    for fo_vec in fo.values():
        active += (np.asarray(fo_vec) > fo_threshold).astype(float)
    return active / n_units


def continuous_occupancy(timecourses, run_boundaries):
    """Soft (non-discretized) occupancy analogue.

    Per-run z-score each component, take per-TR L1-normalized magnitude shares
    (|z_tk| / sum_j |z_tj|), average over all TRs. Hard limit = WTA FO. Sums to
    1 across components. The robustness-arm out-of-stimulus occupancy.
    """
    tc = np.asarray(timecourses, dtype=float)
    z = np.empty_like(tc)
    for (s0, e0) in run_boundaries:
        seg = tc[s0:e0]
        z[s0:e0] = (seg - seg.mean(0)) / (seg.std(0) + 1e-12)
    mag = np.abs(z)
    row = mag.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    shares = mag / row
    return shares.mean(axis=0)
