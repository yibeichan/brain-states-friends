#!/usr/bin/env python3
"""
hmm_io.py - I/O utilities for PCA-projected HMM data.

Handles loading train/valid/test splits, PCA lookup tables, projected run
arrays, and decoding state sequences. Supports primary, LOSO, and split-half
folds via a unified ``fold_spec`` parameter.

fold_spec values:
    None              -> primary split
    'loso_3'          -> LOSO season 3
    'split_half_A'    -> split-half fold A

Legacy: ``loso_season`` parameter is still accepted for backward compatibility
and internally converted to fold_spec='loso_{season}'.
"""

import os
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _fold_subdir(fold_spec):
    """Convert a fold_spec string to a subdirectory path component.

    Returns:
        '' for primary (no subdirectory), or a relative path like
        'loso/season_3' or 'split_half/A'.
    """
    if fold_spec is None:
        return ''
    if fold_spec.startswith('loso_'):
        season = fold_spec[5:]
        return os.path.join('loso', f'season_{season}')
    if fold_spec.startswith('split_half_'):
        half = fold_spec[11:]
        return os.path.join('split_half', half)
    raise ValueError(f"Unknown fold_spec: {fold_spec!r}")


def _resolve_fold_spec(fold_spec=None, loso_season=None):
    """Resolve fold_spec from either fold_spec or legacy loso_season.

    At most one of fold_spec or loso_season may be set.
    """
    if fold_spec is not None and loso_season is not None:
        raise ValueError(
            "Cannot specify both fold_spec and loso_season. "
            "Use fold_spec only (loso_season is deprecated)."
        )
    if loso_season is not None:
        return f'loso_{loso_season}'
    return fold_spec


def load_split(combined_base, split_type='primary', loso_season=None, fold_spec=None):
    """Load split JSON produced by 03a_pca4combined_hmm.

    Primary split:     splits/primary.json
    LOSO fold:         splits/loso_season_{s}.json
    Split-half fold:   splits/split_half_{A|B}.json

    Args:
        combined_base: Root output directory of 03a for this subject.
        split_type:    'primary' or 'loso' (legacy; prefer fold_spec).
        loso_season:   Season integer (legacy; prefer fold_spec='loso_{s}').
        fold_spec:     Unified fold identifier: None, 'loso_3', 'split_half_A'.

    Returns:
        Dict with 'train', 'valid', 'test' run ID lists.
    """
    fs = _resolve_fold_spec(fold_spec, loso_season)

    if fs is None:
        # Primary or legacy split_type routing
        if split_type == 'primary':
            split_path = os.path.join(combined_base, 'splits', 'primary.json')
        elif split_type == 'loso':
            raise ValueError("loso_season or fold_spec required for loso split")
        else:
            raise ValueError(f"Unknown split_type: {split_type!r}")
    elif fs.startswith('loso_'):
        season = fs[5:]
        split_path = os.path.join(
            combined_base, 'splits', f'loso_season_{season}.json'
        )
    elif fs.startswith('split_half_'):
        half = fs[11:]
        split_path = os.path.join(
            combined_base, 'splits', f'split_half_{half}.json'
        )
    else:
        raise ValueError(f"Unknown fold_spec: {fs!r}")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, 'r') as f:
        return json.load(f)


def load_n_pcs_lookup(combined_base, loso_season=None, fold_spec=None):
    """Load n_pcs_lookup from 03a output.

    Primary:         {combined_base}/n_pcs_lookup.json
    LOSO/split-half: {combined_base}/{fold_subdir}/n_pcs_lookup.json

    Each fold has its own PCA fitted on that fold's training data, so
    its n_pcs values may differ from the primary split's values.

    Args:
        combined_base: Root output directory of 03a for this subject.
        loso_season:   Season integer (legacy; prefer fold_spec).
        fold_spec:     Unified fold identifier.

    Returns:
        Dict {"0.80": n, "0.85": n, "0.90": n, ...}
    """
    fs = _resolve_fold_spec(fold_spec, loso_season)
    subdir = _fold_subdir(fs)
    lookup_path = os.path.join(combined_base, subdir, 'n_pcs_lookup.json') if subdir else \
        os.path.join(combined_base, 'n_pcs_lookup.json')
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(f"n_pcs_lookup not found: {lookup_path}")
    with open(lookup_path, 'r') as f:
        return json.load(f)


def get_projected_dir(combined_base, loso_season=None, fold_spec=None):
    """Path to projected data directory.

    Primary:         {combined_base}/projected/
    LOSO/split-half: {combined_base}/{fold_subdir}/projected/

    Args:
        combined_base: Root output directory of 03a for this subject.
        loso_season:   Season integer (legacy; prefer fold_spec).
        fold_spec:     Unified fold identifier.

    Returns:
        Absolute path string to the projected directory.
    """
    fs = _resolve_fold_spec(fold_spec, loso_season)
    subdir = _fold_subdir(fs)
    if subdir:
        return os.path.join(combined_base, subdir, 'projected')
    return os.path.join(combined_base, 'projected')


def load_projected_runs(run_ids, projected_dir, n_pcs, split_name):
    """Load PCA-projected runs from a split subdirectory.

    Files are at: {projected_dir}/{split_name}/{run_id}.npy
    Each file has shape (n_trs, n_pcs_max); truncated to (n_trs, n_pcs).

    Args:
        run_ids:       List of run ID strings.
        projected_dir: Root projected directory (primary or LOSO variant).
        n_pcs:         Number of PCs to keep (slices from the saved n_pcs_max columns).
        split_name:    'train', 'valid', or 'test'.

    Returns:
        X_concat: ndarray (total_trs, n_pcs)
        lengths:  list of int (per-run TR counts)
    """
    arrays, lengths = [], []
    for run_id in run_ids:
        fpath = os.path.join(projected_dir, split_name, f'{run_id}.npy')
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Projected file not found: {fpath}\n"
                f"  (split_name={split_name!r}, run_id={run_id!r})"
            )
        data = np.load(fpath)[:, :n_pcs]
        if data.shape[1] < n_pcs:
            raise ValueError(
                f"Expected {n_pcs} PCs but file has {data.shape[1]} columns: {fpath}"
            )
        if not np.all(np.isfinite(data)):
            n_bad = (~np.isfinite(data)).sum()
            logger.warning(
                f"Run {run_id} ({split_name}): {n_bad} non-finite values, "
                "replacing with 0"
            )
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(data)
        lengths.append(data.shape[0])
    return np.vstack(arrays), lengths


def back_project_states(model, pca, n_pcs):
    """Back-project HMM state parameters from PCA space to parcel space.

    Computes:
        state_means_parcel = model.means_ @ W + pca.mean_
        state_covars_parcel[k] = W.T @ Sigma_pca[k] @ W   (per state)

    where W = pca.components_[:n_pcs] has shape (n_pcs, n_parcels).

    Args:
        model: Fitted StickyHDPHMM with .means_ (K, n_pcs) and
               .covars_ of shape (K, n_pcs, n_pcs) for full or
               (K, n_pcs) for diagonal covariance.
        pca:   Fitted sklearn PCA with .components_ and .mean_.
        n_pcs: Number of PCs used by the model.

    Returns:
        state_means_parcel:  ndarray (K, n_parcels)
        state_covars_parcel: ndarray (K, n_parcels, n_parcels)

    Raises:
        AssertionError: If model.means_.shape[1] != n_pcs.
    """
    assert model.means_.shape[1] == n_pcs, (
        f"model.means_ has {model.means_.shape[1]} columns, expected {n_pcs}"
    )
    W = pca.components_[:n_pcs]  # (n_pcs, n_parcels)

    # --- Means ---
    state_means_parcel = model.means_ @ W + pca.mean_

    # --- Covariances ---
    covars = model.covars_
    K = covars.shape[0]
    n_parcels = W.shape[1]

    if covars.ndim == 3:
        # Full covariance: (K, n_pcs, n_pcs)
        # Sigma_parcel[k] = W.T @ Sigma_pca[k] @ W
        # W shape: (n_pcs, n_parcels), so W.T is (n_parcels, n_pcs)
        # W[p,i] so W.T[i,p] @ Sigma[k,p,q] @ W[q,j] = result[k,i,j]
        state_covars_parcel = np.einsum('pi,kpq,qj->kij', W, covars, W)
    elif covars.ndim == 2:
        # Diagonal covariance: (K, n_pcs)
        # Sigma_pca[k] = diag(covars[k])
        # W.T @ diag(d) @ W = (W.T * d) @ W  — avoids K full diag matrices
        state_covars_parcel = np.zeros((K, n_parcels, n_parcels))
        for k in range(K):
            Wd = W.T * covars[k]  # (n_parcels, n_pcs) * (n_pcs,) broadcast
            state_covars_parcel[k] = Wd @ W
    else:
        raise ValueError(
            f"Unexpected covars_ shape {covars.shape}; expected 2-d or 3-d"
        )

    logger.info(
        "Back-projected states to parcel space: means %s, covars %s",
        state_means_parcel.shape, state_covars_parcel.shape,
    )
    return state_means_parcel, state_covars_parcel


def decode_all_runs(model, split, projected_dir, n_pcs):
    """Decode all runs (train + valid + test) using the fitted model.

    Split-aware: loads each run from its correct subdirectory
    (train/, valid/, or test/) within projected_dir.

    Args:
        model:         Fitted StickyHDPHMM.
        split:         Dict with 'train', 'valid', 'test' run ID lists.
        projected_dir: Base projected directory (primary or LOSO variant).
        n_pcs:         Number of PCs used by this model.

    Returns:
        decoded_states: dict  run_id -> np.array(n_trs,) of state indices
    """
    # Build run -> split_name lookup for file routing
    run_to_split = {}
    for sname in ('train', 'valid', 'test'):
        for run_id in split.get(sname, []):
            run_to_split[run_id] = sname

    decoded_states = {}
    n_failed = 0

    for run_id in sorted(run_to_split.keys()):
        sname = run_to_split[run_id]
        fpath = os.path.join(projected_dir, sname, f'{run_id}.npy')
        if not os.path.exists(fpath):
            logger.error(f"Projected file missing for decode: {fpath}")
            n_failed += 1
            continue

        data = np.load(fpath)[:, :n_pcs]
        if data.shape[1] < n_pcs:
            logger.error(
                f"Expected {n_pcs} PCs but file has {data.shape[1]} columns: {fpath}"
            )
            n_failed += 1
            continue
        if len(data) == 0:
            logger.warning(f"Run {run_id}: empty data array, skipping decode")
            continue
        try:
            _, state_seq = model.decode(data)
            decoded_states[run_id] = state_seq
        except Exception as e:
            logger.error(f"Decode failed for {run_id}: {e}")
            n_failed += 1

    if n_failed > 0:
        logger.warning(
            f"decode_all_runs: {n_failed}/{len(run_to_split)} runs failed"
        )
    return decoded_states
