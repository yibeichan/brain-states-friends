"""Figure F4 Panel C - DINOv2 × Movie10 per-film depth profile (code ID F4).

NB (2026-07 figure swap): manuscript Figures 4 and 5 were reordered in revision.
Code ID "F4"/"fig4" now maps to manuscript Figure 5 (R4b decoding). See
MANIFEST.md "Figure-order note".

For each of the 6 subjects, this script reproduces 08e's Friends-fit /
Movie10-project pipeline for DINOv2-large only, then **breaks the Movie10
balanced-accuracy evaluation into four per-film subsets** (bourne / figures /
life / wolf) instead of pooling across films. Permutation testing is skipped
(point estimates only) - the global significance question was already settled
by the existing 08e D3a output for movie10 × dinov2; here we only need per-
film effect sizes for the poster's Video panel.

Outputs:
  $SCRATCH_DIR/output/manuscript_figures/fig4/perfilm/{sub_id}_dinov2_movie10_perfilm.json
  $SCRATCH_DIR/output/manuscript_figures/fig4/fig4_C_video_perfilm_depth.{pdf,png,svg}
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.common import (
    feature_key_for_cross_stim_run_id, load_training_split,
)
from utils.plot_style import apply_publication_style
from utils.transformer_analysis import (
    INTERSECTION_MIN_FO, build_layer_feature_matrix,
    load_content_eligibility, stream_pca_features,
)
from utils.transformer_io import MODEL_REGISTRY

SCRATCH_DIR = os.environ["SCRATCH_DIR"]
PARCELLATION = "atlas-4S156Parcels"
VT = "vt0.95"
MODEL_KEY = "dinov2-large"
STIM = "movie10"
SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
FILMS = [
    ("wolf",    "Wolf of Wall Street",  "#1f77b4"),
    ("figures", "Hidden Figures",       "#ff7f0e"),
    ("bourne",  "Bourne Supremacy",     "#2ca02c"),
    ("life",    "Life",                 "#d62728"),
]
PCA_VARIANCE_THRESHOLD = 0.95

OUT_DIR = Path(SCRATCH_DIR) / "output" / "manuscript_figures" / "fig4"
PERFILM_DIR = OUT_DIR / "perfilm"
PERFILM_DIR.mkdir(parents=True, exist_ok=True)
apply_publication_style()


def _load_decoded(stage_dir, sub_id):
    base = Path(SCRATCH_DIR) / "output" / stage_dir / PARCELLATION / sub_id
    candidates = [
        base / "final" / VT / "decoded_states.pkl",
        base / "final" / "decoded_states.pkl",
        base / VT / "decoded_states.pkl",
        base / "decoded_states.pkl",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"decoded_states.pkl not found under {base}")


def _compute_intersection(friends_states, test_states, eligible_ids):
    eligible_set = set(int(s) for s in eligible_ids)
    f_counts, t_counts = {}, {}
    for s in friends_states:
        f_counts[int(s)] = f_counts.get(int(s), 0) + 1
    for s in test_states:
        t_counts[int(s)] = t_counts.get(int(s), 0) + 1
    f_total, t_total = len(friends_states), len(test_states)
    kept = []
    for s in sorted(eligible_set):
        f_fo = f_counts.get(s, 0) / max(f_total, 1)
        t_fo = t_counts.get(s, 0) / max(t_total, 1)
        if f_fo >= INTERSECTION_MIN_FO and t_fo >= INTERSECTION_MIN_FO:
            kept.append(s)
    return kept


def run_one_subject(sub_id):
    print(f"\n=== {sub_id} ===")
    friends_decoded = _load_decoded("04_combined_hdphmm", sub_id)
    test_decoded = _load_decoded("m10_04_decoded", sub_id)

    splits = load_training_split(sub_id, PARCELLATION, SCRATCH_DIR)
    friends_run_ids_init = sorted(friends_decoded.keys())
    test_run_ids_init = sorted(test_decoded.keys())
    friends_n_trs_init = {r: len(friends_decoded[r]) for r in friends_run_ids_init}
    test_n_trs_init = {r: len(test_decoded[r]) for r in test_run_ids_init}

    print(f"  loading Friends features + fitting PCA ({MODEL_KEY})...")
    (
        friends_features, pca_info, pca_models, friends_eff, _f_dropped,
    ) = stream_pca_features(
        "friends", MODEL_KEY, friends_run_ids_init, friends_n_trs_init,
        SCRATCH_DIR,
        train_run_ids=splits["train"],
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )

    print(f"  loading Movie10 features + projecting through Friends PCA...")
    (
        test_features, _, _, test_eff, _t_dropped,
    ) = stream_pca_features(
        STIM, MODEL_KEY, test_run_ids_init, test_n_trs_init, SCRATCH_DIR,
        pca_models=pca_models, pca_info=pca_info,
        feature_key_fn=lambda r: feature_key_for_cross_stim_run_id(r, STIM),
        max_tr_drift=3,
    )

    for rid in list(friends_decoded.keys()):
        if rid not in friends_eff:
            friends_decoded.pop(rid); continue
        eff = friends_eff[rid]
        if len(friends_decoded[rid]) != eff:
            friends_decoded[rid] = np.asarray(friends_decoded[rid])[:eff]
    for rid in list(test_decoded.keys()):
        if rid not in test_eff:
            test_decoded.pop(rid); continue
        eff = test_eff[rid]
        if len(test_decoded[rid]) != eff:
            test_decoded[rid] = np.asarray(test_decoded[rid])[:eff]

    friends_run_ids = sorted(friends_decoded.keys())
    test_run_ids = sorted(test_decoded.keys())
    friends_states_cat = np.concatenate([friends_decoded[r] for r in friends_run_ids])
    test_states_cat = np.concatenate([test_decoded[r] for r in test_run_ids])
    test_run_lens = [len(test_decoded[r]) for r in test_run_ids]
    test_tr_run_id = np.concatenate(
        [np.full(L, r) for r, L in zip(test_run_ids, test_run_lens)]
    )

    eligibility = load_content_eligibility(sub_id, PARCELLATION, SCRATCH_DIR, vt="0.95")
    intersect_ids = _compute_intersection(
        friends_states_cat, test_states_cat, eligibility["content_eligible"],
    )
    n_classes = len(intersect_ids)
    if n_classes < 2:
        print(f"  [skip] {sub_id}: intersection < 2 classes")
        return None
    intersect_set = set(intersect_ids)
    friends_mask = np.array([int(s) in intersect_set for s in friends_states_cat], dtype=bool)
    test_mask = np.array([int(s) in intersect_set for s in test_states_cat], dtype=bool)
    chance = 1.0 / n_classes
    print(f"  intersection size n_classes={n_classes}; chance={chance:.4f}")

    film_masks = {}
    for film_key, _, _ in FILMS:
        is_film = np.array([str(r).startswith(film_key) for r in test_tr_run_id])
        film_masks[film_key] = is_film & test_mask
        n_film_trs = int(film_masks[film_key].sum())
        print(f"  film={film_key:<8} n_TRs_in_intersection={n_film_trs}")

    n_layers = MODEL_REGISTRY[MODEL_KEY]["n_layers"]
    per_layer_results = {}
    for layer_idx in range(n_layers):
        friends_runs = friends_features.get(layer_idx, {})
        test_runs = test_features.get(layer_idx, {})
        if not friends_runs or not test_runs:
            continue
        try:
            friends_X = build_layer_feature_matrix(friends_runs, friends_run_ids, friends_decoded)
            test_X = build_layer_feature_matrix(test_runs, test_run_ids, test_decoded)
        except ValueError as exc:
            print(f"  layer {layer_idx}: skipped ({exc})")
            continue

        X_train = friends_X[friends_mask]
        y_train = friends_states_cat[friends_mask].astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred_full = clf.predict(test_X)
        y_true_full = test_states_cat.astype(int)

        per_film = {}
        for film_key, _, _ in FILMS:
            m = film_masks[film_key]
            if not m.any():
                continue
            y_t, y_p = y_true_full[m], y_pred_full[m]
            if len(np.unique(y_t)) < 2:
                continue
            ba = float(balanced_accuracy_score(y_t, y_p))
            per_film[film_key] = {
                "balanced_accuracy": round(ba, 4),
                "chance_level": round(chance, 4),
                "delta": round(ba - chance, 4),
                "n_test_trs": int(m.sum()),
                "n_test_classes_present": int(len(np.unique(y_t))),
            }
        per_layer_results[layer_idx] = per_film

    out_payload = {
        "sub_id": sub_id, "model": MODEL_KEY, "stimulus": STIM,
        "n_classes_intersection": n_classes,
        "chance_level": round(chance, 4),
        "per_layer": {str(k): v for k, v in per_layer_results.items()},
    }
    out_path = PERFILM_DIR / f"{sub_id}_{MODEL_KEY}_{STIM}_perfilm.json"
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"  wrote {out_path.name}")
    return out_payload


def render_panel():
    print("\n=== rendering Panel C (Video × per-film) ===")
    n_layers = MODEL_REGISTRY[MODEL_KEY]["n_layers"]
    curves = {f[0]: np.full((len(SUBJECTS), n_layers), np.nan) for f in FILMS}
    for s_idx, sub_id in enumerate(SUBJECTS):
        p = PERFILM_DIR / f"{sub_id}_{MODEL_KEY}_{STIM}_perfilm.json"
        if not p.exists():
            print(f"  [skip] no result for {sub_id}")
            continue
        d = json.loads(p.read_text())
        for L_str, per_film in d["per_layer"].items():
            L = int(L_str)
            for film_key in curves:
                if film_key in per_film:
                    curves[film_key][s_idx, L] = per_film[film_key]["delta"]

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    xs = np.arange(n_layers)
    for film_key, film_label, color in FILMS:
        arr = curves[film_key]
        n_per_layer = np.sum(~np.isnan(arr), axis=0).clip(min=1)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_per_layer)
        n_subj = int(np.max(n_per_layer))
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, mean, color=color, linewidth=1.6, label=f"{film_label} (n={n_subj})")
        peak = int(np.nanargmax(mean))
        print(f"  {film_key:<8} n={n_subj}  peak layer={peak}/{n_layers-1}  "
              f"peak Δ={mean[peak]:.4f}  mean Δ={np.nanmean(mean):.4f}")
    ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_xlabel("DINOv2-large layer", fontsize=7)
    ax.set_ylabel("Balanced accuracy − chance", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_pdf = OUT_DIR / "fig4_C_video_perfilm_depth.pdf"
    out_png = OUT_DIR / "fig4_C_video_perfilm_depth.png"
    out_svg = OUT_DIR / "fig4_C_video_perfilm_depth.svg"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> wrote {out_pdf.name} + {out_png.name}")


def main():
    print(f"=== Fig F4 Panel C - per-film video depth profile ===")
    print(f"  OUT_DIR = {OUT_DIR}")
    for sub_id in SUBJECTS:
        try:
            run_one_subject(sub_id)
        except Exception as e:
            print(f"  ERROR processing {sub_id}: {e}")
            import traceback
            traceback.print_exc()
    render_panel()
    print("done.")


if __name__ == "__main__":
    main()
