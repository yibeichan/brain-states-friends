"""Core functions for 06e_transition_modules (R3 transition modules).

Graphs are dense weight matrices W (n x n, float) over a fixed, sorted list of
state ids; W[i, j] > 0 means an edge i -> j with that weight. Self-loops are
never present. networkx graphs are built only where Louvain needs them, always
with nodes inserted in sorted order so seeded Louvain is reproducible.

All randomness comes from a numpy Generator passed in by the caller.
"""
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def weight_matrix(P, nodes, threshold=None, renorm_rows=False):
    """Off-diagonal weights of P restricted to `nodes` (in that order).

    threshold: keep entries >= threshold (06b's comparator); None keeps all.
    renorm_rows: rescale each row's off-diagonal mass (over `nodes`) to 1
    before thresholding ("where does the state go when it leaves").
    """
    idx = np.asarray(nodes, dtype=int)
    W = np.array(P, dtype=float)[np.ix_(idx, idx)]
    np.fill_diagonal(W, 0.0)
    if renorm_rows:
        rs = W.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            W = np.where(rs > 0, W / rs, 0.0)
    if threshold is not None:
        W = np.where(W >= threshold, W, 0.0)
    return W


def to_digraph(W):
    G = nx.DiGraph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = np.nonzero(W)
    G.add_weighted_edges_from(
        (int(i), int(j), float(W[i, j])) for i, j in zip(rows, cols))
    return G


def run_transition_counts(decoded_by_run, K):
    """Per-run transition count matrices (K x K), within runs only."""
    out = {}
    for run_id, seq in decoded_by_run.items():
        seq = np.asarray(seq, dtype=int)
        C = np.zeros((K, K))
        if seq.size > 1:
            np.add.at(C, (seq[:-1], seq[1:]), 1.0)
        out[run_id] = C
    return out


def counts_to_P(C):
    """Row-normalise over all K (self-transitions included), as 06b does."""
    rs = C.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rs > 0, C / rs, 0.0)


# --------------------------------------------------------------------------
# Modularity
# --------------------------------------------------------------------------

def labels_from_communities(comms, n):
    lab = np.full(n, -1, dtype=int)
    # order modules by their smallest member so labels are deterministic
    for k, members in enumerate(sorted(comms, key=lambda c: min(c))):
        for m in members:
            lab[m] = k
    return lab


def best_modularity(W, seeds, resolution=1.0):
    """Best directed weighted modularity over Louvain runs with fixed seeds.

    Returns (Q_best, labels_best, share_of_seeds_at_best).
    """
    G = to_digraph(W)
    if G.number_of_edges() == 0:
        return 0.0, np.zeros(W.shape[0], dtype=int), 1.0
    qs, best = [], None
    for s in seeds:
        comms = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=int(s))
        q = nx.community.modularity(G, comms, weight="weight", resolution=resolution)
        qs.append(q)
        if best is None or q > best[0] + 1e-12:
            best = (q, comms)
    qs = np.array(qs)
    share = float(np.mean(np.isclose(qs, best[0], atol=1e-9)))
    return float(best[0]), labels_from_communities(best[1], W.shape[0]), share


def modularity_of(W, labels, resolution=1.0):
    G = to_digraph(W)
    comms = [set(np.flatnonzero(labels == k).tolist()) for k in np.unique(labels)]
    return float(nx.community.modularity(G, comms, weight="weight", resolution=resolution))


def reciprocity(W):
    A = W > 0
    n_edges = A.sum()
    return float((A & A.T).sum() / n_edges) if n_edges else np.nan


# --------------------------------------------------------------------------
# N1: degree-, reciprocity- and strength-preserving rewiring
# --------------------------------------------------------------------------

def _split_dyads(A):
    """Reciprocal dyads (i < j, both directions) and single directed edges."""
    R = A & A.T
    iu = np.argwhere(np.triu(R, 1))
    single = np.argwhere(A & ~A.T)
    return [tuple(x) for x in iu], [tuple(x) for x in single]


def rewire_topology(A, rng, swaps_per_edge=10, max_tries_factor=50):
    """Directed Maslov-Sneppen swaps that keep in/out degree and reciprocity.

    Single (non-reciprocated) edges swap with single edges, (a->b, c->d) ->
    (a->d, c->b); reciprocal dyads swap with dyads, {a,b},{c,d} -> {a,d},{c,b}.
    A swap is rejected if it would create a self-loop, a duplicate, or any
    edge between a new pair that already has an edge in either direction
    (which would change reciprocity). Returns (A_new, n_accepted, n_tried).
    """
    A = A.copy().astype(bool)
    dyads, single = _split_dyads(A)
    dyads = [list(d) for d in dyads]
    single = [list(e) for e in single]
    n_edges = int(A.sum())
    target = swaps_per_edge * n_edges
    accepted = tried = 0
    max_tries = max_tries_factor * max(target, 1)
    n_d, n_s = len(dyads), len(single)
    while accepted < target and tried < max_tries:
        tried += 1
        use_dyad = n_d >= 2 and (n_s < 2 or rng.random() < 2 * n_d / (2 * n_d + n_s))
        pool = dyads if use_dyad else single
        if len(pool) < 2:
            continue
        x, y = rng.choice(len(pool), size=2, replace=False)
        a, b = pool[x]
        c, d = pool[y]
        if use_dyad and rng.random() < 0.5:
            c, d = d, c  # dyads are unordered: try both pairings
        if a == d or c == b or a == c or b == d:
            continue
        if A[a, d] or A[d, a] or A[c, b] or A[b, c]:
            continue
        if use_dyad:
            A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
            A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
            pool[x] = [a, d]
            pool[y] = [c, b]
        else:
            A[a, b] = A[c, d] = False
            A[a, d] = A[c, b] = True
            pool[x] = [a, d]
            pool[y] = [c, b]
        accepted += 1
    return A, accepted, tried


def assign_weights_strength_matched(A_new, W_obs, rng):
    """Rubinov & Sporns (2011)-style weight reassignment on a fixed topology.

    Observed weights are placed on the new edges so that out/in strengths are
    approximately preserved: repeatedly rank the remaining edges by expected
    weight s_out_res[i] * s_in_res[j] and the remaining weights by size, pick a
    random rank, give that rank's weight to that rank's edge, and update the
    residual strengths. Total weight is preserved exactly.
    """
    weights = np.sort(W_obs[W_obs > 0])
    s_out = W_obs.sum(axis=1).astype(float)
    s_in = W_obs.sum(axis=0).astype(float)
    edges = np.argwhere(A_new)
    W = np.zeros_like(W_obs, dtype=float)
    remaining = np.ones(len(edges), dtype=bool)
    w_left = list(weights)
    eps = 1e-12
    for _ in range(len(edges)):
        idx = np.flatnonzero(remaining)
        exp_w = np.maximum(s_out[edges[idx, 0]], eps) * np.maximum(s_in[edges[idx, 1]], eps)
        order = idx[np.argsort(exp_w, kind="stable")]
        r = int(rng.integers(len(order)))
        e = order[r]
        w = w_left.pop(r)
        i, j = edges[e]
        W[i, j] = w
        s_out[i] -= w
        s_in[j] -= w
        remaining[e] = False
    return W


def rewire_null(W_obs, rng, swaps_per_edge=10):
    A_new, acc, tried = rewire_topology(W_obs > 0, rng, swaps_per_edge)
    W = assign_weights_strength_matched(A_new, W_obs, rng)
    return W, {"accepted": acc, "tried": tried}


def rewire_null_permuted_weights(W_obs, rng, swaps_per_edge=10):
    """Degree- and reciprocity-preserving swaps, then the observed weights
    permuted at random over the new edges (strengths are not preserved)."""
    A_new, acc, tried = rewire_topology(W_obs > 0, rng, swaps_per_edge)
    W = np.zeros_like(W_obs, dtype=float)
    W[A_new] = rng.permutation(W_obs[W_obs > 0])
    return W, {"accepted": acc, "tried": tried}


def strength_error(W_null, W_obs):
    """Mean absolute relative error of out- and in-strengths."""
    errs = []
    for ax in (1, 0):
        so, sn = W_obs.sum(axis=ax), W_null.sum(axis=ax)
        ok = so > 0
        errs.append(np.abs(sn[ok] - so[ok]) / so[ok])
    return float(np.mean(np.concatenate(errs)))


def row_shuffle_null(W_full, rng):
    """N1.row: permute each row's off-diagonal entries (keeps out-strength)."""
    n = W_full.shape[0]
    W = np.zeros_like(W_full)
    for i in range(n):
        off = np.array([j for j in range(n) if j != i])
        W[i, off] = rng.permutation(W_full[i, off])
    return W


# --------------------------------------------------------------------------
# N2: segment shuffle within runs (keeps which states occur where, and dwell)
# --------------------------------------------------------------------------

def run_segments(seq):
    seq = np.asarray(seq, dtype=int)
    if seq.size == 0:
        return np.array([], int), np.array([], int)
    change = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [seq.size]]))
    return seq[starts], lengths


def segment_tables(decoded_by_run):
    return {r: run_segments(s) for r, s in decoded_by_run.items()}


def shuffled_counts(seg_tables, K, rng):
    """Transition counts after permuting segment order within every run.

    Off-diagonal counts come from consecutive segment labels that differ;
    each state's row total is its TR count minus one if it ends the run (the
    same totals as the unshuffled data, apart from which segment is last), so
    self-transitions absorb the rest, including merges of adjacent equal
    labels created by the shuffle.
    """
    C = np.zeros((K, K))
    for labels, lengths in seg_tables.values():
        if labels.size == 0:
            continue
        perm = rng.permutation(labels.size)
        lab, ln = labels[perm], lengths[perm]
        a, b = lab[:-1], lab[1:]
        diff = a != b
        np.add.at(C, (a[diff], b[diff]), 1.0)
        tot = np.bincount(lab, weights=ln, minlength=K)
        tot[lab[-1]] -= 1
        off = np.bincount(a[diff], minlength=K)
        self_ = tot - off
        C[np.arange(K), np.arange(K)] += self_
    return C


# --------------------------------------------------------------------------
# N3: non-modular chain with fitted stickiness and popularity
# --------------------------------------------------------------------------

def nonmodular_chain(P_model, nodes):
    """P_ij = (1 - p_ii) q_j / (1 - q_i) on `nodes`; q from in-strength."""
    idx = np.asarray(nodes, dtype=int)
    Psub = np.array(P_model, dtype=float)[np.ix_(idx, idx)]
    p_stay = np.clip(np.diag(Psub) / Psub.sum(axis=1), 0, 1 - 1e-9)
    off = Psub.copy()
    np.fill_diagonal(off, 0.0)
    q = off.sum(axis=0)
    q = q / q.sum()
    n = len(idx)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i] = (1 - p_stay[i]) * q / (1 - q[i])
        Q[i, i] = p_stay[i]
    return Q, p_stay, q


def simulate_chain_sequences(Q, run_lengths, start_states, rng):
    """State sequences (one per run, labels 0..n-1) simulated from Q."""
    n = Q.shape[0]
    cum = np.cumsum(Q, axis=1)
    out = {}
    for k, (L, s0) in enumerate(zip(run_lengths, start_states)):
        u = rng.random(L - 1)
        seq = np.empty(L, dtype=int)
        seq[0] = s = int(s0)
        for t in range(L - 1):
            s = min(int(np.searchsorted(cum[s], u[t] * cum[s, -1], side="right")), n - 1)
            seq[t + 1] = s
        out[f"run{k:03d}"] = seq
    return out


def modular_chain(Q_base, labels, ratio):
    """Off-diagonal mass reweighted by `ratio` within modules, rows renormalised
    so each state's stay probability is unchanged."""
    Q = Q_base.copy()
    stay = np.diag(Q).copy()
    off = Q - np.diag(stay)
    same = labels[:, None] == labels[None, :]
    off = off * np.where(same, ratio, 1.0)
    off = off / off.sum(axis=1, keepdims=True) * (1 - stay)[:, None]
    return off + np.diag(stay)


def simulate_chain_counts(Q, run_lengths, start_states, rng):
    """Transition counts (n x n) from simulating Q with given run lengths."""
    n = Q.shape[0]
    cum = np.cumsum(Q, axis=1)
    C = np.zeros((n, n))
    for L, s0 in zip(run_lengths, start_states):
        u = rng.random(L - 1)
        s = int(s0)
        for t in range(L - 1):
            nxt = int(np.searchsorted(cum[s], u[t] * cum[s, -1], side="right"))
            nxt = min(nxt, n - 1)
            C[s, nxt] += 1
            s = nxt
    return C


# --------------------------------------------------------------------------
# Similarity-predicted null (test a)
# --------------------------------------------------------------------------

def predicted_weight_model(W_full, predictors):
    """OLS of log off-diagonal P on dyadic predictors + log strengths.

    predictors: dict name -> (n x n) matrix (diagonal ignored). Pairs with a
    non-finite predictor or zero P are excluded from the fit. Returns the
    fitted log-prediction for every off-diagonal pair, the residuals of the
    fitted pairs, the fit mask, and the coefficients.
    """
    n = W_full.shape[0]
    off = ~np.eye(n, dtype=bool)
    s_out = W_full.sum(axis=1)
    s_in = W_full.sum(axis=0)
    cols = {"log_s_out": np.log(np.maximum(s_out, 1e-12))[:, None] * np.ones((1, n)),
            "log_s_in": np.ones((n, 1)) * np.log(np.maximum(s_in, 1e-12))[None, :]}
    cols.update(predictors)
    names = list(cols)
    X = np.stack([cols[k][off] for k in names], axis=1)
    y = np.log(np.where(W_full[off] > 0, W_full[off], np.nan))
    fit = np.isfinite(X).all(axis=1) & np.isfinite(y)
    Xd = np.column_stack([np.ones(fit.sum()), X[fit]])
    beta, *_ = np.linalg.lstsq(Xd, y[fit], rcond=None)
    pred = np.full(off.sum(), np.nan)
    ok = np.isfinite(X).all(axis=1)
    pred[ok] = np.column_stack([np.ones(ok.sum()), X[ok]]) @ beta
    resid = y[fit] - Xd @ beta
    r2 = 1 - resid.var() / y[fit].var()
    return {"pred": pred, "resid": resid, "fit_mask": fit, "ok_mask": ok,
            "beta": dict(zip(["intercept"] + names, beta.tolist())), "r2": float(r2),
            "off_mask": off}


def predicted_null_graph(model, n, threshold, rng):
    """exp(prediction + residuals permuted across dyads), thresholded."""
    off = model["off_mask"]
    vals = np.full(off.sum(), 0.0)
    ok = model["ok_mask"]
    res = rng.choice(model["resid"], size=ok.sum(), replace=True) \
        if ok.sum() != model["resid"].size else rng.permutation(model["resid"])
    vals[ok] = np.exp(model["pred"][ok] + res)
    W = np.zeros((n, n))
    W[off] = vals
    return np.where(W >= threshold, W, 0.0)


# --------------------------------------------------------------------------
# Stability: consensus clustering (Lancichinetti & Fortunato 2012)
# --------------------------------------------------------------------------

def coassignment(label_sets):
    L = np.asarray(label_sets)
    return (L[:, :, None] == L[:, None, :]).mean(axis=0)


def consensus_partition(D, rng, tau=0.5, n_runs=100, max_iter=20):
    D = D.copy()
    for it in range(max_iter):
        Dt = np.where(D >= tau, D, 0.0)
        np.fill_diagonal(Dt, 0.0)
        G = nx.Graph()
        G.add_nodes_from(range(D.shape[0]))
        r, c = np.nonzero(np.triu(Dt, 1))
        G.add_weighted_edges_from((int(i), int(j), float(Dt[i, j])) for i, j in zip(r, c))
        labs = []
        for _ in range(n_runs):
            comms = nx.community.louvain_communities(
                G, weight="weight", seed=int(rng.integers(2**31 - 1)))
            labs.append(labels_from_communities(comms, D.shape[0]))
        D = coassignment(labs)
        if np.all((D == 0) | (D == 1)):
            return labs[0], it + 1, True
    return labs[0], max_iter, False


# --------------------------------------------------------------------------
# Module timescale and stimulus locking
# --------------------------------------------------------------------------

def module_dwell(decoded_by_run, module_of_state):
    """Mean length of consecutive TRs within one module (states outside the
    graph get module -1 and break dwell)."""
    lens = []
    for seq in decoded_by_run.values():
        m = module_of_state[np.asarray(seq, dtype=int)]
        _, ln = run_segments(m)
        labs, _ = run_segments(m)
        lens.append(ln[labs >= 0])
    lens = np.concatenate(lens)
    return float(lens.mean()) if lens.size else np.nan


def module_dwell_shuffled(seg_tables, module_of_state, rng):
    lens = []
    for labels, lengths in seg_tables.values():
        if labels.size == 0:
            continue
        perm = rng.permutation(labels.size)
        m = module_of_state[labels[perm]]
        ln = lengths[perm]
        change = np.flatnonzero(np.diff(m)) + 1
        starts = np.concatenate([[0], change])
        grp = np.add.reduceat(ln, starts)
        lens.append(grp[m[starts] >= 0])
    lens = np.concatenate(lens)
    return float(lens.mean()) if lens.size else np.nan


def match_modules(centroids_a, centroids_b):
    """Hungarian match of module centroid maps on 1 - r (signed)."""
    A = centroids_a - centroids_a.mean(1, keepdims=True)
    B = centroids_b - centroids_b.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    R = A @ B.T
    ia, ib = linear_sum_assignment(1 - R)
    return ia, ib, R[ia, ib]


def occupancy_locking(mod_a_by_run, mod_b_by_run, pairs, runs):
    """Mean over matched module pairs of r(occupancy_a, occupancy_b),
    concatenated over common runs (truncated to the shorter run)."""
    xs = {k: [] for k in range(len(pairs))}
    ys = {k: [] for k in range(len(pairs))}
    for r in runs:
        a, b = mod_a_by_run[r], mod_b_by_run[r]
        L = min(a.size, b.size)
        for k, (ma, mb) in enumerate(pairs):
            xs[k].append(a[:L] == ma)
            ys[k].append(b[:L] == mb)
    rs = []
    for k in range(len(pairs)):
        x = np.concatenate(xs[k]).astype(float)
        y = np.concatenate(ys[k]).astype(float)
        if x.std() == 0 or y.std() == 0:
            rs.append(np.nan)
        else:
            rs.append(float(np.corrcoef(x, y)[0, 1]))
    return np.array(rs)


# --------------------------------------------------------------------------
# Dyadic predictors and MRQAP (descriptive characterisation)
# --------------------------------------------------------------------------

def bhattacharyya_diag(mu, var):
    """Pairwise Bhattacharyya distance between diagonal Gaussians."""
    n = mu.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        v = (var[i] + var) / 2.0
        dm = mu[i] - mu
        term1 = 0.125 * np.sum(dm * dm / v, axis=1)
        term2 = 0.5 * (np.sum(np.log(v), axis=1)
                       - 0.5 * (np.sum(np.log(var[i])) + np.sum(np.log(var), axis=1)))
        D[i] = term1 + term2
    return D


def mahalanobis_diag(mu, var):
    n = mu.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        v = (var[i] + var) / 2.0
        dm = mu[i] - mu
        D[i] = np.sqrt(np.sum(dm * dm / v, axis=1))
    return D


def mrqap_dekker(Y, preds, rng, n_perm=5000):
    """MRQAP with Dekker semi-partialling on symmetric dyadic data (i < j).

    Y: (n x n) outcome; preds: dict name -> (n x n). Returns per-predictor
    coefficient, t, and two-sided permutation p (node relabelling of that
    predictor's residual matrix).
    """
    n = Y.shape[0]
    iu = np.triu_indices(n, 1)
    names = list(preds)
    X = np.column_stack([preds[k][iu] for k in names])
    y = Y[iu].astype(float)

    def fit_t(Xm):
        Xd = np.column_stack([np.ones(len(y)), Xm])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        dof = len(y) - Xd.shape[1]
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.pinv(Xd.T @ Xd)
        return beta[1:], beta[1:] / np.sqrt(np.diag(cov)[1:])

    beta, t_obs = fit_t(X)
    out = {}
    for k, name in enumerate(names):
        others = np.delete(X, k, axis=1)
        Xo = np.column_stack([np.ones(len(y)), others])
        g, *_ = np.linalg.lstsq(Xo, X[:, k], rcond=None)
        e = X[:, k] - Xo @ g
        E = np.zeros((n, n))
        E[iu] = e
        E = E + E.T
        t_null = np.empty(n_perm)
        for p in range(n_perm):
            perm = rng.permutation(n)
            Xp = X.copy()
            Xp[:, k] = E[np.ix_(perm, perm)][iu]
            t_null[p] = fit_t(Xp)[1][k]
        pval = float((1 + np.sum(np.abs(t_null) >= abs(t_obs[k]))) / (1 + n_perm))
        out[name] = {"beta": float(beta[k]), "t": float(t_obs[k]), "p": pval}
    return out
