"""Unit tests for 05h_participant_state_matching.

Claims tested: (a) state mixing keeps the mean map and the parcel-space
scatter and changes the states; the configuration rotation keeps the Gram
matrix and the span; (b) the PC-block rotation rotates, and keeps each state's
norm within every block; (c) a planted correspondence beats the state-mixing
null; (d) with no correspondence but a shared dominant subspace the
state-mixing null stays calibrated and the configuration rotation does not;
(e) joint-count occupancy r equals the direct Pearson r of binary occupancy,
family-wise locking flags are stricter than pointwise ones, and trimming
keeps the runs aligned; (f) network collapse and component removal behave as
defined; (g) Haar draws are sign-corrected; (h) the provenance record refuses
uncommitted code. Synthetic data only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import importlib
m = importlib.import_module("05h_participant_state_matching")


def _subject_maps(rng, k=20, P=156, n_pcs=60, shared=None, noise=0.5):
    basis = np.linalg.qr(rng.normal(size=(P, n_pcs)))[0].T        # (n_pcs, P)
    spec = np.linspace(3, 0.2, n_pcs)
    mu = rng.normal(size=(k, n_pcs)) * spec
    maps = mu @ basis
    if shared is not None:
        maps[: shared.shape[0]] = shared + noise * rng.normal(size=shared.shape)
    return maps, mu, basis


def test_configuration_rotation_keeps_gram_and_span():
    rng = np.random.default_rng(0)
    M, _, _ = _subject_maps(rng)
    R = m.rotate_configuration(M, rng)
    C, Cr = M - M.mean(0), R - R.mean(0)
    assert np.allclose(C @ C.T, Cr @ Cr.T, atol=1e-8)
    # same span: projecting Cr onto span(C) changes nothing
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    assert np.allclose(Cr @ Vt.T @ Vt, Cr, atol=1e-8)
    assert not np.allclose(Cr, C)


def test_state_mixing_keeps_mean_and_scatter():
    rng = np.random.default_rng(10)
    M, _, _ = _subject_maps(rng, k=15)
    M = M + 3.0                                                   # non-zero mean map
    X = m.mix_states(M, rng)
    C, Cx = M - M.mean(0), X - X.mean(0)
    assert np.allclose(X.mean(0), M.mean(0), atol=1e-10)
    assert np.allclose(C.T @ C, Cx.T @ Cx, atol=1e-8)
    assert not np.allclose(X, M)


def test_pc_block_rotation_keeps_block_norms():
    rng = np.random.default_rng(1)
    M, mu, basis = _subject_maps(rng)
    R = m.rotate_pc_blocks(mu, basis, rng, block=5)
    assert not np.allclose(R, mu @ basis)
    mu_r = R @ basis.T                                            # back to PC coords
    for s in range(0, mu.shape[1], 5):
        assert np.allclose(np.linalg.norm(mu[:, s:s + 5], axis=1),
                           np.linalg.norm(mu_r[:, s:s + 5], axis=1))


def test_planted_correspondence_beats_state_mixing_null():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=(12, 156))
    A, _, _ = _subject_maps(rng, shared=shared)
    B, _, _ = _subject_maps(rng, shared=shared)
    r = m.match(m.corr_matrix(A, B))[2]
    null = m.pair_null(A, B, 200, rng, m.NULL)[0]
    assert m.p_greater(np.median(r), null) < 0.05


def _shared_subspace_rejection_rate(kind, n_pairs=60, seed=3):
    """No state correspondence; both repertoires live mostly in one shared
    8-dim subspace (sd 3) plus 25 private dimensions (sd 1)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_pairs):
        W = np.linalg.qr(rng.normal(size=(156, 8)))[0].T
        A, B = (m._sim_subject(W, k, 3.0, 25, 1.0, rng) for k in (20, 18))
        r = m.match(m.corr_matrix(A, B))[2]
        null = m.pair_null(A, B, 99, rng, kind)[0]
        hits += m.p_greater(np.median(r), null) < 0.05
    return hits / n_pairs


def test_state_mixing_null_calibrated_with_shared_subspace():
    assert _shared_subspace_rejection_rate(m.NULL) <= 0.12


def test_configuration_rotation_rejects_shared_subspace():
    # why it is no longer the primary null
    assert _shared_subspace_rejection_rate("configuration", n_pairs=20) >= 0.5


def test_occupancy_r_matches_direct_pearson():
    rng = np.random.default_rng(4)
    la = {"r1": rng.integers(-1, 3, 50), "r2": rng.integers(-1, 3, 40)}
    lb = {"r1": rng.integers(-1, 4, 55), "r2": rng.integers(-1, 4, 40)}
    runs = ["r1", "r2"]
    counts = m.joint_counts(la, lb, runs, 3, 4)
    T = 50 + 40
    got = m.occupancy_r(counts, T, [(0, 1), (2, 3)])
    xa = np.concatenate([la["r1"][:50], la["r2"]])
    xb = np.concatenate([lb["r1"][:50], lb["r2"]])
    for (a, b), g in zip([(0, 1), (2, 3)], got):
        assert g == pytest.approx(np.corrcoef(xa == a, xb == b)[0, 1])


def test_collapse_and_project_out():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(4, 6))
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    C = m.collapse_networks(M, groups)
    assert np.allclose(C[:, 0], C[:, 1]) and np.allclose(C[:, 2], M[:, 2:5].mean(1))
    V = m.shared_components([rng.normal(size=(10, 6))], 2)
    P = m.project_out(M, V)
    assert np.allclose(P @ V.T, 0, atol=1e-10)


def test_many_to_one_counts_collisions():
    R = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.7]])
    best, coll = m.many_to_one(R)
    assert coll == 1 and best.tolist() == [0.9, 0.8, 0.7]


def test_p_values_handle_nan():
    assert np.isnan(m.p_greater(np.nan, np.arange(10.0)))
    null = np.array([0.1, np.nan, 0.3, np.nan])
    assert m.p_greater(0.2, null) == pytest.approx((1 + 1) / (1 + 2))
    assert m.p_less(0.2, null) == pytest.approx((1 + 1) / (1 + 2))


def test_label_map_does_not_need_sorted_states():
    seq = np.array([4, 2, 9, 4, 7])
    out = m.label_map(seq, [9, 4, 2], 10)
    assert out.tolist() == [1, 2, 0, 1, -1]


def _toy_participants(rng, locked=True, n_runs=8, L=300):
    """Two participants with 3 states each; state 0 of A and 0 of B co-occur."""
    da = {"decoded": {}, "K": 3}
    db = {"decoded": {}, "K": 3}
    for r in range(n_runs):
        base = np.repeat(rng.integers(0, 3, L // 10), 10)
        other = np.repeat(rng.integers(0, 3, L // 10), 10)
        da["decoded"][f"r{r}"] = base
        db["decoded"][f"r{r}"] = base.copy() if locked else other
    return da, db


def test_locking_flags_planted_pair_and_not_shuffled():
    rng = np.random.default_rng(3)
    da, db = _toy_participants(rng, locked=True)
    runs = sorted(da["decoded"])
    obs, null, n_runs, T = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 200, rng)
    r, thr, fwer, pointwise = m.lock_flags(obs, null, np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert all(fwer) and r.min() > 0.9 and n_runs == 8
    da, db = _toy_participants(rng, locked=False)
    obs, null, _, _ = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 200, rng)
    _, _, fwer, _ = m.lock_flags(obs, null, np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert not any(fwer)


def test_locking_trim_drops_run_edges():
    rng = np.random.default_rng(4)
    da, db = _toy_participants(rng, locked=True, n_runs=2, L=100)
    runs = sorted(da["decoded"])
    obs, _, _, T = m.locking_matrices(da, db, [0, 1, 2], [0, 1, 2], runs, 5, rng, trim=10)
    assert T == 2 * (100 - 20)
    assert np.diag(obs).min() > 0.9                               # both sides trimmed alike


def test_family_wise_lock_flag_is_stricter_than_pointwise():
    rng = np.random.default_rng(6)
    null = rng.normal(size=(20000, 3, 3))
    obs = np.zeros((3, 3))
    obs[0, 0] = 1.9          # above its own 95th pct (~1.64), below the max-of-3 one (~2.12)
    idx = np.arange(3)
    _, thr, fwer, pointwise = m.lock_flags(obs, null, idx, idx)
    assert pointwise[0] and not fwer[0] and thr > 2.0


def test_haar_is_sign_corrected():
    rng = np.random.default_rng(7)
    q00 = np.array([m.haar(3, rng)[0, 0] for _ in range(4000)])
    assert abs(q00.mean()) < 0.05                                 # unfixed QR gives q00 < 0 always


def test_provenance_refuses_uncommitted_or_unknown_code(tmp_path, monkeypatch):
    f = tmp_path / "x.npy"
    f.write_bytes(b"abc")
    dseg = tmp_path / "dseg.tsv"
    dseg.write_text("index\tlabel\n")
    data = {"sub-01": {"input_files": [str(f)]}}

    class Args:
        parcellation, vt, n_perm, n_lock, seed = "p", "0.95", 10, 10, 0
    monkeypatch.setattr(m, "dseg_path", lambda parc: str(dseg))
    monkeypatch.setattr(m, "git_head", lambda: "abc123")
    monkeypatch.setattr(m, "git_dirty_scripts", lambda: " M script/05h_x.py")
    with pytest.raises(SystemExit):
        m.provenance_record(Args, data)
    monkeypatch.setattr(m, "git_dirty_scripts", lambda: None)
    with pytest.raises(SystemExit):
        m.provenance_record(Args, data)
    monkeypatch.setattr(m, "git_dirty_scripts", lambda: "")
    rec = m.provenance_record(Args, data)
    assert rec["code_commit"] == "abc123" and set(rec["inputs_sha256"]) == {"sub-01", "dseg"}
    before = rec["inputs_sha256"]["sub-01"]
    f.write_bytes(b"abd")
    assert m.provenance_record(Args, data)["inputs_sha256"]["sub-01"] != before


def test_cosine_keeps_amplitude_sign():
    A = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]])
    B = np.array([[-2.0, -2.0, -2.0]])
    C = m.cos_matrix(A, B)
    assert C[0, 0] == pytest.approx(-1.0) and C[1, 0] == pytest.approx(0.0)


def test_scope_maps_slices_then_centres():
    M = np.arange(12.0).reshape(3, 4)
    X = m.scope_maps(M, np.array([True, False, True, True]), True)
    assert X.shape == (3, 3) and np.allclose(X.mean(0), 0)
    assert np.allclose(m.scope_maps(M, slice(None), False), M)


def test_cortex_mask_comes_from_the_dseg(tmp_path, monkeypatch):
    # cortical parcels need not come first
    rows = ["index\tlabel\tnetwork_label", "1\tLH-Pu\tn/a", "2\tLH_Vis_1\tVis",
            "3\tCerebellar_Region1\tn/a", "4\tRH_Default_pCunPCC_1\tDefault"]
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "p_dseg.tsv").write_text("\n".join(rows) + "\n")
    monkeypatch.setenv("ATLAS_DIR", str(tmp_path))
    groups, cortex = m.parcel_groups("p")
    assert cortex.tolist() == [False, True, False, True]
    assert groups[1] == "Vis" and groups[3] == "Default"


def _toy_subject(rng, sub, k=8, P=12, n_pcs=6, runs=4, L=120):
    comps = np.linalg.qr(rng.normal(size=(P, P)))[0].T
    mu_pc = rng.normal(size=(k, n_pcs)) * np.linspace(2, 0.5, n_pcs)
    decoded = {f"r{r}": np.repeat(rng.integers(0, k, L // 6), 6) for r in range(runs)}
    states = list(range(k))
    return {"sub": sub, "maps": mu_pc @ comps[:n_pcs], "mu_pc": mu_pc, "comps": comps,
            "decoded": decoded, "K": k, "active": states, "eligible": states[:6],
            "onset": [], "classes": {6: "D", 7: "B"},
            "flags": __import__("pandas").DataFrame(
                {"summary_category": ["eligible_for_content_analysis"] * 6 + ["rare"] * 2}),
            "rec": rng.random(k), "fo": rng.dirichlet(np.ones(k)), "dwell": rng.random(k) + 1}


def test_analyse_pair_runs_every_variant_with_both_nulls():
    rng = np.random.default_rng(8)
    subs = [_toy_subject(rng, f"sub-0{i}") for i in range(1, 7)]
    groups = np.array(list("aabbccddeeff"))
    cortex = np.array([True] * 8 + [False] * 4)

    class Args:
        n_perm, n_lock, seed = 20, 20, 0
    res = m.analyse_pair(subs[0], subs[1], subs[2:], groups, cortex, Args)
    keys = {f"{s}|{sc}|{c}" for s in ("eligible", "active") for sc in ("full", "cortex")
            for c in ("raw", "centred")}
    assert keys <= set(res["variants"])
    for k in keys:
        v = res["variants"][k]
        assert np.isfinite(v[m.NULL]["p"]) and np.isfinite(v["pc_blocks"]["p"])
        assert v["n_matched"] == (6 if k.startswith("eligible") else 8)
        assert all("beyond_max_null95" not in x for x in v["matches_full"])
    assert "validation" in res["variants"]["eligible|full|raw"]
