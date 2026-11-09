#!/usr/bin/env python3
"""04s_sweep_config_final.py - build a final/vt{vt}/ directory from a saved sweep fit.

The hyperparameter-sensitivity analysis reruns the downstream pipeline on the
train-split fits saved by `04_combined_hdphmm.py --mode fit` (one directory per
config under configs/, five seeds each) instead of refitting. This script
picks one seed of one config and writes the same files that
`04_combined_hdphmm.select_finalize` writes for its chosen refit model, using
the same helpers (utils.hmm_io.decode_all_runs, back_project_states,
load_projected_runs; utils.hdphmm.infer_n_active_states):

    best_model.pkl, decoded_states.pkl, state_means_pca.npy,
    state_means_parcel.npy, state_covars.npy, state_covars_parcel.npy,
    pca_model.pkl, final_results.json

final_results.json keeps select_finalize's keys (downstream scripts read
model_info.n_states, data_info.n_pcs, final_refit.n_active_states and
final_refit.test_ll_per_sample) and adds a `sweep_source` block naming the
config directory, seed and seed rule.

Seed rule (`best_valid`): the seed with the highest validation LL per sample
among successful seeds. It must equal config_summary.json's
best_seed_by_valid; a mismatch is an error.

Inputs are read from --source_scratch (the published dataset) for the sweep
fit, and from $SCRATCH_DIR/output/03a_pca4combined_hmm for the split, the
projected runs and the PCA. Output goes under $SCRATCH_DIR/output. With
--protect, the script refuses to write anywhere that resolves inside that
directory.

Usage:
    SCRATCH_DIR=<side root> python script/04s_sweep_config_final.py \\
        --sub_id sub-01 --config vt0.95_covdiag_nc50_g1 \\
        --source_scratch /path/to/published --protect /path/to/published/output

    # Gate: rebuild from the published final model instead of a sweep seed
    SCRATCH_DIR=<side root> python script/04s_sweep_config_final.py \\
        --sub_id sub-06 --from_final /path/to/published/output/.../final/vt0.95
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import _get_season  # noqa: E402
from utils.hmm_io import (  # noqa: E402
    back_project_states, decode_all_runs, get_projected_dir, load_n_pcs_lookup,
    load_projected_runs, load_split,
)

logger = logging.getLogger("04s_sweep_config_final")

SCHEMA_INFO = {'schema_version': '3.1', 'baseline_primary': 'active_states',
               'writer': '04s_sweep_config_final'}


def choose_seed(summary, rule='best_valid'):
    """Seed chosen by `rule` from a config_summary.json dict.

    Returns (seed, seed_result). Raises ValueError if no seed succeeded or the
    argmax disagrees with the summary's own best_seed_by_valid.
    """
    if rule != 'best_valid':
        raise ValueError(f"unknown seed rule {rule!r}")
    ok = [r for r in summary['seed_results'] if r.get('status') == 'success'
          and r.get('valid_ll_per_sample') is not None]
    if not ok:
        raise ValueError(f"{summary.get('config_name')}: no successful seeds")
    best = max(ok, key=lambda r: r['valid_ll_per_sample'])
    vals = [r['valid_ll_per_sample'] for r in ok]
    if vals.count(best['valid_ll_per_sample']) > 1:
        raise ValueError(f"{summary.get('config_name')}: tied best validation LL")
    if summary.get('best_seed_by_valid') != best['seed']:
        raise ValueError(
            f"{summary.get('config_name')}: argmax valid LL seed {best['seed']} != "
            f"config_summary best_seed_by_valid {summary.get('best_seed_by_valid')}")
    return int(best['seed']), best


def state_covars_for_save(covars, covariance_type):
    """Covariances as the production final/ stores them for this covariance type.

    The JAX model (every production refit) keeps diagonal covariances as
    (K, D). The numpy WeakLimitHMM (some sweep fits, e.g. nc40/nc80) returns
    them from its covars_ property expanded to (K, D, D) diagonal matrices.
    06e reads state_covars.npy and requires (K, D) for a diagonal model, so a
    (K, D, D) array of a diagonal model is reduced to its diagonals after
    checking that every off-diagonal entry is exactly zero.
    """
    covars = np.asarray(covars)
    if covariance_type != 'diag' or covars.ndim == 2:
        return covars
    if covars.ndim != 3 or covars.shape[1] != covars.shape[2]:
        raise ValueError(f"unexpected covariance shape {covars.shape}")
    diag = np.diagonal(covars, axis1=1, axis2=2).copy()
    off = covars.copy()
    idx = np.arange(covars.shape[1])
    off[:, idx, idx] = 0.0
    if np.any(off != 0):
        raise ValueError("diag model with nonzero off-diagonal covariances")
    return diag


def resolves_inside(path, protected):
    """True if `path` (existing or not) resolves to `protected` or below it."""
    prot = os.path.realpath(protected)
    p = os.path.abspath(path)
    # resolve the nearest existing ancestor, then re-append the rest
    tail = []
    while not os.path.exists(p):
        p, t = os.path.split(p)
        tail.append(t)
    real = os.path.join(os.path.realpath(p), *reversed(tail))
    return real == prot or real.startswith(prot.rstrip('/') + '/')


def final_dir_for(scratch_dir, parcellation, sub_id, vt):
    return os.path.join(scratch_dir, 'output', '04_combined_hdphmm', parcellation,
                        sub_id, 'final', f'vt{vt:.2f}')


def build_final(model, config, config_name, sub_id, parcellation, combined_base,
                final_dir, source_info, protect=None):
    """Write select_finalize's final/ files for `model` into `final_dir`."""
    from utils.hdphmm import infer_n_active_states

    if protect and resolves_inside(final_dir, protect):
        raise RuntimeError(f"refusing to write {final_dir}: resolves inside {protect}")
    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    n_pcs = n_pcs_lookup[f"{config['variance_threshold']:.2f}"]
    if model.means_.shape[1] != n_pcs:
        raise RuntimeError(f"model has {model.means_.shape[1]} PCs, lookup says {n_pcs}")
    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)

    X_test, lengths_test = load_projected_runs(split['test'], projected_dir, n_pcs, 'test')
    n_test = sum(lengths_test)
    test_ll_ps = model.score(X_test, lengths=lengths_test) / n_test
    n_states = int(model.n_components)
    n_active_states = max(1, infer_n_active_states(
        model, min_state_usage=config['min_state_usage']))
    base_total = np.log(1.0 / n_states)
    base_active = np.log(1.0 / n_active_states)
    test_by_season = {}
    for r in split['test']:
        test_by_season.setdefault(_get_season(r), []).append(r)
    test_ll_per_season = {}
    for s, runs in sorted(test_by_season.items()):
        X_st, l_st = load_projected_runs(runs, projected_dir, n_pcs, 'test')
        test_ll_per_season[f's{s}'] = float(model.score(X_st, lengths=l_st) / sum(l_st))

    logger.info("Decoding all episodes...")
    decoded_states = decode_all_runs(model, split, projected_dir, n_pcs)

    with open(os.path.join(combined_base, 'pca_model.pkl'), 'rb') as f:
        pca = pickle.load(f)
    means_parcel, covars_parcel = back_project_states(model, pca, n_pcs)

    os.makedirs(final_dir, exist_ok=True)
    if protect and resolves_inside(final_dir, protect):
        raise RuntimeError(f"refusing to write {final_dir}: resolves inside {protect}")
    with open(os.path.join(final_dir, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f, protocol=4)
    with open(os.path.join(final_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f, protocol=4)
    with open(os.path.join(final_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded_states, f, protocol=4)
    np.save(os.path.join(final_dir, 'state_means_pca.npy'), model.means_)
    np.save(os.path.join(final_dir, 'state_means_parcel.npy'), means_parcel)
    np.save(os.path.join(final_dir, 'state_covars_parcel.npy'), covars_parcel)
    np.save(os.path.join(final_dir, 'state_covars.npy'),
            state_covars_for_save(model.covars_, config['covariance_type']))

    lengths_trainval = [len(np.load(os.path.join(projected_dir, sname, f'{r}.npy'),
                                    mmap_mode='r'))
                        for sname in ('train', 'valid') for r in split[sname]]
    final_result = {
        'sub_id': sub_id, 'parcellation': parcellation,
        'selected_config': config, 'selected_config_name': config_name,
        'selection_metric': None,
        'all_config_rankings': [],
        'final_refit': {
            'n_seeds': source_info.get('n_seeds'), 'best_seed': source_info.get('seed'),
            'trainval_ll_per_sample': source_info.get('trainval_ll_per_sample'),
            'test_ll_per_sample': float(test_ll_ps),
            'test_ll_per_season': test_ll_per_season,
            'baseline_ll_per_sample_total_states': float(base_total),
            'baseline_ll_per_sample_active_states': float(base_active),
            'generalizes_vs_baseline_total_states': bool(test_ll_ps > base_total),
            'generalizes_vs_baseline_active_states': bool(test_ll_ps > base_active),
            'n_total_states': n_states, 'n_active_states': int(n_active_states),
            'overfit_gap': None,
            'seed_results': source_info.get('seed_results', []),
        },
        'data_info': {
            'n_train_runs': len(split['train']), 'n_valid_runs': len(split['valid']),
            'n_test_runs': len(split['test']),
            'n_trainval_samples': int(sum(lengths_trainval)),
            'n_test_samples': int(n_test), 'n_pcs': int(n_pcs),
            'variance_threshold': config['variance_threshold'],
        },
        'model_info': {'n_states': n_states},
        'model_selection': {'method': 'sweep_config_seed', 'select_config_override': config_name},
        'sweep_source': source_info,
        'schema_info': dict(SCHEMA_INFO),
        'n_decoded_runs': len(decoded_states),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(final_dir, 'final_results.json'), 'w') as f:
        json.dump(final_result, f, indent=2)
    logger.info("Final results saved to %s (%d states, %d K_used, %d runs, test LL/TR %.4f)",
                final_dir, n_states, n_active_states, len(decoded_states), test_ll_ps)
    return final_result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sub_id', required=True)
    ap.add_argument('--parcellation', default='atlas-4S156Parcels')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--config', help='sweep config directory name under configs/')
    src.add_argument('--from_final', help='an existing final/vt{vt} directory (gate)')
    ap.add_argument('--seed_rule', default='best_valid', choices=['best_valid'])
    ap.add_argument('--source_scratch', default=None,
                    help='SCRATCH root holding output/04_combined_hdphmm/.../configs '
                         '(default: $SCRATCH_DIR)')
    ap.add_argument('--protect', default=None,
                    help='refuse to write anywhere that resolves inside this directory')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing final_results.json')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    scratch = os.environ.get('SCRATCH_DIR')
    if not scratch:
        raise SystemExit("SCRATCH_DIR is not set")
    combined_base = os.path.join(scratch, 'output', '03a_pca4combined_hmm',
                                 args.parcellation, args.sub_id)

    if args.config:
        source_scratch = args.source_scratch or scratch
        cfg_dir = os.path.join(source_scratch, 'output', '04_combined_hdphmm',
                               args.parcellation, args.sub_id, 'configs', args.config)
        with open(os.path.join(cfg_dir, 'config_summary.json')) as f:
            summary = json.load(f)
        if summary['config_name'] != args.config:
            raise SystemExit(f"config_summary names {summary['config_name']}, not {args.config}")
        seed, seed_res = choose_seed(summary, args.seed_rule)
        model_path = os.path.join(cfg_dir, f'seed_{seed}_model.pkl')
        config, config_name = summary['config'], summary['config_name']
        if summary.get('n_pcs') is not None:
            lookup = load_n_pcs_lookup(combined_base)
            if lookup[f"{config['variance_threshold']:.2f}"] != summary['n_pcs']:
                raise SystemExit("config_summary n_pcs differs from 03a n_pcs_lookup")
        source_info = {
            'kind': 'sweep_config', 'config_dir': cfg_dir,
            'model_path': model_path, 'model_realpath': os.path.realpath(model_path),
            'seed_rule': args.seed_rule, 'seed': seed, 'n_seeds': summary['n_seeds_total'],
            'fit_split': 'train (primary split); validation used only for seed choice',
            'valid_ll_per_sample': seed_res['valid_ll_per_sample'],
            'train_ll_per_sample': seed_res['train_ll_per_sample'],
            'trainval_ll_per_sample': None,
            'converged': seed_res.get('converged'),
            'n_iterations': seed_res.get('n_iterations'),
            'n_active_states_fit': seed_res.get('n_active_states'),
            'seed_results': summary['seed_results'],
        }
    else:
        with open(os.path.join(args.from_final, 'final_results.json')) as f:
            fr = json.load(f)
        model_path = os.path.join(args.from_final, 'best_model.pkl')
        config, config_name = fr['selected_config'], fr['selected_config_name']
        source_info = {
            'kind': 'existing_final', 'final_dir': args.from_final,
            'model_path': model_path, 'model_realpath': os.path.realpath(model_path),
            'seed': fr['final_refit']['best_seed'], 'n_seeds': fr['final_refit']['n_seeds'],
            'trainval_ll_per_sample': fr['final_refit']['trainval_ll_per_sample'],
            'seed_results': fr['final_refit']['seed_results'],
        }

    final_dir = final_dir_for(scratch, args.parcellation, args.sub_id,
                              config['variance_threshold'])
    if os.path.exists(os.path.join(final_dir, 'final_results.json')) and not args.force:
        logger.info("Final results exist at %s. Use --force to redo.", final_dir)
        return
    logger.info("Loading %s", model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    build_final(model, config, config_name, args.sub_id, args.parcellation, combined_base,
                final_dir, source_info, protect=args.protect)


if __name__ == '__main__':
    main()
