"""Compare the states of two HMM fits of the same runs.

Used by clean_wmcsf_compare.py to ask whether replacing the corrupt-mask CSF/WM
regressors with clean ones moves the fitted states more than refitting with another
seed does. A fit is a dict with `means` (n_states, n_parcels) in parcel space,
`decoded` (run id -> state per TR) and `k_used` (states the model uses).

Metrics, for fit A against fit B:
- median_r: median spatial Pearson r over Hungarian-matched active states;
- ari: adjusted Rand index of the per-TR labels over all runs (needs no matching);
- mapped_agreement: among TRs in matched A states, the fraction whose B state is
  the matched one;
- fo_distance: total-variation distance between pooled occupancies after mapping,
  with each fit's unmatched states pooled into one bin.

The verdict and the permutation test use median_r, ari and fo_distance
(VERDICT_METRICS); the other metrics are reported only. Both raise on a
non-finite verdict metric instead of reading it as "not moved" or p = 0.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from utils.recurrence_utils import active_state_indices

# Direction of "worse" for each verdict metric.
VERDICT_METRICS = {"median_r": "lower", "ari": "lower", "fo_distance": "higher"}


def _row_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / np.where(np.linalg.norm(a, axis=1, keepdims=True) > 0,
                     np.linalg.norm(a, axis=1, keepdims=True), 1.0)
    b = b / np.where(np.linalg.norm(b, axis=1, keepdims=True) > 0,
                     np.linalg.norm(b, axis=1, keepdims=True), 1.0)
    return a @ b.T


def match_states(means_a, means_b, active_a, active_b):
    """Hungarian match of active states on 1 - spatial correlation.

    Returns the A -> B state mapping and the matched correlations, in A order.
    """
    active_a, active_b = np.asarray(active_a), np.asarray(active_b)
    if active_a.size == 0 or active_b.size == 0:
        return {"mapping": {}, "correlations": np.array([]), "n_matched": 0}
    corr = _row_corr(means_a[active_a], means_b[active_b])
    rows, cols = linear_sum_assignment(1.0 - corr)
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    mapping = {int(active_a[r]): int(active_b[c]) for r, c in zip(rows, cols)}
    return {"mapping": mapping, "correlations": corr[rows, cols],
            "n_matched": len(mapping)}


def _concat(decoded_a, decoded_b):
    if set(decoded_a) != set(decoded_b):
        raise ValueError("the two fits decoded different runs")
    keys = sorted(decoded_a)
    for k in keys:
        if len(decoded_a[k]) != len(decoded_b[k]):
            raise ValueError(f"run {k}: {len(decoded_a[k])} vs {len(decoded_b[k])} TRs")
    return (np.concatenate([np.asarray(decoded_a[k]) for k in keys]),
            np.concatenate([np.asarray(decoded_b[k]) for k in keys]))


def label_agreement(decoded_a, decoded_b, mapping):
    """ARI over all TRs, and per-TR agreement over TRs whose A state is matched.

    `unmatched_fraction` is the share of TRs in A states left out of `mapping`.
    """
    a, b = _concat(decoded_a, decoded_b)
    mapped = np.array([mapping.get(int(s), -1) for s in a])
    in_matched = mapped >= 0
    agreement = float(np.mean(mapped[in_matched] == b[in_matched])) if in_matched.any() \
        else float("nan")
    return {"ari": float(adjusted_rand_score(a, b)),
            "mapped_agreement": agreement,
            "unmatched_fraction": float(1.0 - in_matched.mean()),
            "n_trs": int(a.size)}


def pooled_fo(decoded, n_states):
    """Fraction of all TRs, pooled over runs, spent in each state."""
    labels = np.concatenate([np.asarray(v) for v in decoded.values()])
    return np.bincount(labels, minlength=n_states) / labels.size


def fo_distance(fo_a, fo_b, mapping):
    """Total-variation distance between occupancies with B relabelled into A.

    Matched states are compared pairwise. The states left unmatched on each side
    (inactive or surplus states) are pooled into one "other" bin per fit, so two
    identical fits are at distance 0.
    """
    fo_a, fo_b = np.asarray(fo_a, dtype=float), np.asarray(fo_b, dtype=float)
    a_idx = np.array(list(mapping.keys()), dtype=int)
    b_idx = np.array(list(mapping.values()), dtype=int)
    matched = np.abs(fo_a[a_idx] - fo_b[b_idx]).sum() if a_idx.size else 0.0
    other_a = fo_a.sum() - fo_a[a_idx].sum()
    other_b = fo_b.sum() - fo_b[b_idx].sum()
    return float(0.5 * (matched + abs(other_a - other_b)))


def compare_fits(fit_a, fit_b):
    """All metrics for fit A against fit B (active states: FO > 0.02 in a run)."""
    n_a, n_b = len(fit_a["means"]), len(fit_b["means"])
    act_a = active_state_indices(fit_a["decoded"], n_a)
    act_b = active_state_indices(fit_b["decoded"], n_b)
    m = match_states(fit_a["means"], fit_b["means"], act_a, act_b)
    lab = label_agreement(fit_a["decoded"], fit_b["decoded"], m["mapping"])
    fo = fo_distance(pooled_fo(fit_a["decoded"], n_a), pooled_fo(fit_b["decoded"], n_b),
                     m["mapping"])
    r = m["correlations"]
    return {
        "median_r": float(np.median(r)) if r.size else float("nan"),
        "min_r": float(r.min()) if r.size else float("nan"),
        "n_matched": m["n_matched"],
        "n_active_a": int(act_a.size), "n_active_b": int(act_b.size),
        "k_used_a": int(fit_a["k_used"]), "k_used_b": int(fit_b["k_used"]),
        "ari": lab["ari"], "mapped_agreement": lab["mapped_agreement"],
        "unmatched_fraction": lab["unmatched_fraction"],
        "fo_distance": fo,
    }


def moved_verdict(cross, within):
    """Does the cross-variant comparison fall outside every within-variant pair?

    `cross` is the metrics dict for the two selected fits; `within` is a list of
    metrics dicts for seed pairs inside each variant. A metric is outside when it
    is worse than the worst within-variant pair; the states moved if any is.
    """
    if not within:
        raise ValueError("need at least one within-variant seed pair")
    by_metric = {}
    for name, worse in VERDICT_METRICS.items():
        vals = np.array([w[name] for w in within], dtype=float)
        if not np.isfinite(cross[name]):
            raise ValueError(f"{name}: the cross-variant value is {cross[name]}")
        if not np.isfinite(vals).all():
            raise ValueError(f"{name}: {int((~np.isfinite(vals)).sum())} of {vals.size} "
                             "within-variant seed pairs are not finite")
        entry = {"cross": float(cross[name]),
                 "within_min": float(vals.min()),
                 "within_median": float(np.median(vals)),
                 "within_max": float(vals.max())}
        if worse == "lower":
            entry["outside"] = bool(cross[name] < entry["within_min"])
        else:
            entry["outside"] = bool(cross[name] > entry["within_max"])
        by_metric[name] = entry
    return {"moved": any(e["outside"] for e in by_metric.values()),
            "rule": "moved if any metric is worse than every within-variant seed pair",
            "by_metric": by_metric}


def _split_statistics(values, iu, splits, worse):
    """Within-minus-cross median (signed so larger = cross worse) per split."""
    same = splits[:, iu[0]] == splits[:, iu[1]]            # (n_splits, n_pairs)
    within = np.where(same, values, np.nan)
    cross = np.where(~same, values, np.nan)
    w_med, c_med = np.nanmedian(within, axis=1), np.nanmedian(cross, axis=1)
    return (w_med - c_med) if worse == "lower" else (c_med - w_med), w_med, c_med


def distribution_test(pair_metrics, labels, max_exact=200_000, n_random=20_000,
                      seed=0, chunk=5_000):
    """Seed-level permutation test: are cross-variant pairs worse than within?

    `pair_metrics` maps every pair (i, j), i < j, of the pooled seed fits to its
    metrics dict; `labels` gives each fit's variant (0/1). The statistic is the
    median over within-variant pairs minus the median over cross-variant pairs,
    signed so that larger means the cross-variant pairs look worse. The null
    reassigns the variant labels over whole fits, keeping group sizes, which
    respects pairs that share a seed. All splits are enumerated when there are at
    most `max_exact`; otherwise `n_random` random splits plus the observed one,
    the same splits for every metric. p is the share of splits whose statistic is
    at least the observed one.
    """
    from itertools import combinations
    from math import comb

    labels = np.asarray(labels)
    n = labels.size
    n1 = int((labels == labels[0]).sum())
    iu = np.triu_indices(n, k=1)
    missing = [(int(i), int(j)) for i, j in zip(*iu) if (int(i), int(j)) not in pair_metrics]
    if missing:
        raise ValueError(f"pair_metrics lacks {len(missing)} pairs, e.g. {missing[0]}")

    observed = (labels == labels[0]).astype(np.int8)
    n_total = comb(n, n1)
    if n_total <= max_exact:
        def gen():
            for idx in combinations(range(n), n1):
                row = np.zeros(n, dtype=np.int8)
                row[list(idx)] = 1
                yield row
        exact = True
        n_splits = n_total
    else:
        rng = np.random.default_rng(seed)
        sampled = np.zeros((n_random + 1, n), dtype=np.int8)
        sampled[0] = observed
        for row in sampled[1:]:
            row[rng.choice(n, size=n1, replace=False)] = 1

        def gen():
            yield from sampled
        exact = False
        n_splits = n_random + 1

    by_metric = {}
    for name, worse in VERDICT_METRICS.items():
        values = np.array([pair_metrics[(int(i), int(j))][name] for i, j in zip(*iu)],
                          dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: {int((~np.isfinite(values)).sum())} of "
                             f"{values.size} seed pairs are not finite")
        obs, w_med, c_med = _split_statistics(values, iu, observed[None, :], worse)
        n_ge = 0
        rows = gen()
        while True:
            batch = [r for _, r in zip(range(chunk), rows)]
            if not batch:
                break
            stats, _, _ = _split_statistics(values, iu, np.stack(batch), worse)
            n_ge += int(np.sum(stats >= obs[0] - 1e-12))
        by_metric[name] = {"statistic": float(obs[0]),
                           "within_median": float(w_med[0]),
                           "cross_median": float(c_med[0]),
                           "p": n_ge / n_splits}
    return {"n_splits": n_splits, "exact": exact,
            "statistic": "median(within) - median(cross), signed so larger = cross worse",
            "by_metric": by_metric}
