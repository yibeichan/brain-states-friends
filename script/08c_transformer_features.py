#!/usr/bin/env python3
"""
08c_transformer_features.py - Extract layer-wise transformer features per run.

Extracts hidden-state activations from every layer of a frozen pretrained
transformer, aggregated to TR resolution.  Features are stimulus-level
(same audio/video/text for all subjects), so output is stored without
sub_id in the path.  PCA reduction happens at analysis time (08d) using
subject-specific training splits.

Models (TRIBEv2-validated backbones, frozen, all layers):
  Audio: Wav2VecBert 2.0  (24 layers × 1024 dim) — Friends, M10, PP
  Video: DINOv2-large     (24 layers × 1024 dim) — Friends, M10
  Text:  LLaMA 3.2 3B     (28 layers × 3072 dim) — Friends, M10, HP, PP

Prerequisites:
    uv sync --extra torch
    For audio: ffmpeg installed, mkv stimulus files accessible
    For text: transcript files (algonauts TSV or events.tsv)
    For video: mkv stimulus files accessible

Outputs:
    {SCRATCH_DIR}/output/08c_transformer_features/{stimulus}/{model}/
        layer_00/{run_id}_raw.npy   # (n_trs, hidden_dim), float32
        layer_01/{run_id}_raw.npy
        ...
        metadata.json               # model info, run list, n_trs per run
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.transformer_io import (
    MODEL_REGISTRY,
    TR_DURATION,
    load_model,
    extract_audio_features,
    extract_video_features,
    extract_text_features,
    extract_audio_from_video,
    load_transcript_friends_m10,
    load_transcript_hp,
    load_transcript_pp,
    save_extraction_metadata,
    save_layer_features,
)

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
DATA_DIR = os.getenv("DATA_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")
if DATA_DIR is None:
    raise ValueError("DATA_DIR must be set in the .env file")

ALGONAUTS_DIR = os.getenv("ALGONAUTS_DIR")
if ALGONAUTS_DIR is None:
    raise ValueError("ALGONAUTS_DIR must be set in the .env file")
ALGONAUTS_DIR = os.path.join(ALGONAUTS_DIR, "stimuli")
# DATA_DIR points at the all_about_cneuromod root (see .env.example).
PP_DATASET_DIR = os.path.join(DATA_DIR, "petit-prince")
PP_STIMULI_DIR = os.path.join(PP_DATASET_DIR, "stimuli")
# PP word-level annotations (downloaded from OpenNeuro ds003643 via direct
# HTTPS since the dataset's s3-PUBLIC annex remote has broken credentials).
PP_ANNOTATION_DIR = os.path.join(SCRATCH_DIR, "output", "pp_annotations")
CNEUROMOD_FRIENDS_DIR = os.path.join(DATA_DIR, "cneuromod", "friends")
CNEUROMOD_M10_DIR = os.path.join(DATA_DIR, "cneuromod", "movie10")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Stimulus configuration ────────────────────────────────────────────────

STIMULUS_CONFIG = {
    "friends": {
        # Movie files now come from the authoritative cneuromod dataset.
        # Algonauts transcripts remain the text source (cneuromod has none).
        "movies_dir": os.path.join(CNEUROMOD_FRIENDS_DIR, "stimuli"),
        "transcript_dir": os.path.join(ALGONAUTS_DIR, "transcripts/friends"),
        "modalities": ["audio", "video", "text"],
        "output_subdir": "friends",
        "transcript_type": "tr_aligned_tsv",
    },
    "movie10": {
        # Movie files now come from the authoritative cneuromod dataset.
        # Run-to-clip mapping is read from task-*_events.tsv in `events_dir`.
        # Algonauts transcripts remain the text source (cneuromod has none).
        "movies_dir": os.path.join(CNEUROMOD_M10_DIR, "stimuli"),
        "events_dir": CNEUROMOD_M10_DIR,
        "transcript_dir": os.path.join(ALGONAUTS_DIR, "transcripts/movie10"),
        "modalities": ["audio", "video", "text"],
        "output_subdir": "movie10",
        "transcript_type": "tr_aligned_tsv",
    },
    "harrypotter": {
        "events_dir": os.path.join(DATA_DIR, "cneuromod/harrypotter"),
        "modalities": ["text"],
        "output_subdir": "harrypotter",
        "transcript_type": "hp_events_tsv",
    },
    "petitprince_fr": {
        "stimuli_dir": PP_STIMULI_DIR,
        "events_dir": PP_DATASET_DIR,
        "transcript_path": os.path.join(
            PP_ANNOTATION_DIR, "FR", "lppFR_word_information.csv"
        ),
        "modalities": ["audio", "text"],
        "output_subdir": "petitprince_fr",
        "transcript_type": "pp_transcript",
        "language": "fr",
        "task_prefix": "lppFR",
        "wav_pattern": "task-lppFR_section_{section}.wav",
    },
    "petitprince_en": {
        "stimuli_dir": PP_STIMULI_DIR,
        "events_dir": PP_DATASET_DIR,
        "transcript_path": os.path.join(
            PP_ANNOTATION_DIR, "EN", "lppEN_word_information.csv"
        ),
        "modalities": ["audio", "text"],
        "output_subdir": "petitprince_en",
        "transcript_type": "pp_transcript",
        "language": "en",
        "task_prefix": "lppEN",
        "wav_pattern": "task-lppEN_section-{section}.wav",
    },
}

# Mapping from model key to required modality
MODEL_MODALITY = {
    "w2v-bert-2.0": "audio",
    "dinov2-large": "video",
    "llama-3.2-3b": "text",
}


# ── Run discovery ─────────────────────────────────────────────────────────


def discover_runs_friends(config):
    """Discover Friends runs from transcript directory.

    Returns:
        list of dict: [{run_id, transcript_path, video_path, n_trs}, ...]
    """
    runs = []
    transcript_dir = Path(config["transcript_dir"])

    for season_dir in sorted(transcript_dir.iterdir()):
        if not season_dir.is_dir():
            continue
        for tsv_path in sorted(season_dir.glob("friends_*.tsv")):
            # Extract run_id from filename: friends_s01e01a.tsv -> s01e01a
            stem = tsv_path.stem  # friends_s01e01a
            run_id = stem.replace("friends_", "")

            # Skip season 7 (no fMRI data)
            if run_id.startswith("s07"):
                continue

            # Count TRs from transcript rows
        
            df = pd.read_csv(tsv_path, sep="\t")
            n_trs = len(df)

            # Find matching video
            season_num = run_id[1:3]  # "01" from "s01e01a"
            video_dir = Path(config["movies_dir"]) / f"s{int(season_num)}"
            video_candidates = list(video_dir.glob(f"*{run_id}*.mkv"))
            video_path = video_candidates[0] if video_candidates else None

            runs.append({
                "run_id": run_id,
                "transcript_path": str(tsv_path),
                "video_path": str(video_path) if video_path else None,
                "n_trs": n_trs,
            })

    return runs


def discover_runs_movie10(config):
    """Discover Movie10 runs from cneuromod events.tsv files.

    Each ``task-{run_id}_events.tsv`` has a single row whose ``stim_file``
    column points to ``{genre}/{clip}.mkv`` relative to ``movies_dir``.
    ``run_id`` is the clip name (e.g. ``bourne01``), matching the existing
    08c output layout. Algonauts TR-aligned transcripts are still used for
    the text modality since cneuromod has none.

    ``n_trs`` is computed as ``ceil((onset + duration) / TR_DURATION)`` —
    this is an asymmetrically-safe overshoot (08d truncates features to
    ``len(decoded_states[run_id])``; an undershoot would silently drop the
    run). Matches the HP/PP discovery convention.

    Returns:
        list of dict: [{run_id, video_path, transcript_path, n_trs}, ...]
    """
    events_dir = Path(config["events_dir"])
    movies_dir = Path(config["movies_dir"])
    transcript_dir = Path(config["transcript_dir"])

    runs = []
    for events_path in sorted(events_dir.glob("task-*_events.tsv")):
        # task-bourne01_events.tsv -> bourne01  (anchored strip, not substring)
        run_id = events_path.stem.removeprefix("task-").removesuffix("_events")

        df = pd.read_csv(events_path, sep="\t")
        if df.empty or not {"onset", "duration", "stim_file"}.issubset(df.columns):
            logger.warning(
                "Skipping %s: empty or missing required columns", events_path.name
            )
            continue
        if len(df) > 1:
            logger.warning(
                "Expected 1 row in %s, got %d — using first",
                events_path.name, len(df),
            )

        row = df.iloc[0]
        stim_rel = Path(str(row["stim_file"]))  # e.g. Path("bourne/bourne01.mkv")
        if stim_rel.is_absolute():
            logger.warning(
                "stim_file is absolute, ignoring movies_dir: %s", stim_rel
            )
            video_path = stim_rel
        else:
            video_path = movies_dir / stim_rel
        if not video_path.exists():
            logger.warning("Video not found (datalad get?): %s", video_path)
            video_path = None

        genre = stim_rel.parts[0] if stim_rel.parts else ""

        end_time = float(row["onset"]) + float(row["duration"])
        n_trs = int(np.ceil(end_time / TR_DURATION))

        # Optional transcript from Algonauts (same layout as before)
        transcript_path = transcript_dir / genre / f"movie10_{run_id}.tsv"
        transcript_path_str = (
            str(transcript_path) if transcript_path.exists() else None
        )

        runs.append({
            "run_id": run_id,
            "video_path": str(video_path) if video_path else None,
            "transcript_path": transcript_path_str,
            "n_trs": n_trs,
        })

    return runs


def discover_runs_hp(config):
    """Discover HP runs from events.tsv files.

    Uses sub-01 events as reference (word presentation is the same for all subjects).

    Returns:
        list of dict: [{run_id, events_path, n_trs}, ...]
    """


    runs = []
    events_dir = Path(config["events_dir"])

    # Use sub-01 as reference subject for events.
    # HP may have session dirs (sub-01/ses-*/func) or flat layout (sub-01/func).
    func_dirs = sorted(events_dir.glob("sub-01/*/func"))
    if not func_dirs:
        # Flat layout — no session directories
        flat = events_dir / "sub-01" / "func"
        if flat.is_dir():
            func_dirs = [flat]
    for func_dir in func_dirs:
        for events_path in sorted(func_dir.glob("*_task-harrypotter_*_events.tsv")):
            # Extract run number: sub-01_task-harrypotter_run-01_events.tsv
            stem = events_path.stem
            run_match = None
            for part in stem.split("_"):
                if part.startswith("run-"):
                    run_match = part
                    break

            run_id = f"harrypotter_{run_match}" if run_match else stem

            # Determine n_trs: use last word onset + word duration + buffer.
            # Downstream 08d truncates to min(features, states), so slight
            # overestimate is safe; underestimate loses data.
            df = pd.read_csv(events_path, sep="\t")
            last_row = df.iloc[-1]
            end_time = last_row["onset"] + last_row.get("duration", 0.5)
            n_trs = int(np.ceil(end_time / TR_DURATION))

            runs.append({
                "run_id": run_id,
                "events_path": str(events_path),
                "n_trs": n_trs,
            })

    return runs


def discover_runs_pp(config):
    """Discover Petit Prince runs from WAV files in the stimuli directory.

    PP is audiobook listening — WAV files are the direct stimuli (no video).
    n_trs is derived from audio duration.  Onset offset (pre-stimulus silence
    in the fMRI scan) is read live from each run's events.tsv and stored as
    ``onset_s`` so downstream text alignment can use the exact scan-time
    formula ``tr_idx = int((onset_s + word_onset) / TR_DURATION)`` without
    compound-flooring errors.

    Returns:
        list of dict: [{run_id, audio_path, n_trs, audio_trs, onset_trs,
                        onset_s, section, transcript_path}, ...]
    """
    import wave

    stimuli_dir = Path(config["stimuli_dir"])
    events_dir = Path(config["events_dir"])
    wav_pattern = config["wav_pattern"]
    task_prefix = config["task_prefix"]
    transcript_path_str = config.get("transcript_path")
    transcript_path = Path(transcript_path_str) if transcript_path_str else None
    n_sections = 9

    runs = []
    for section in range(1, n_sections + 1):
        wav_name = wav_pattern.format(section=section)
        wav_path = stimuli_dir / wav_name
        run_id = f"{task_prefix}_run-{section:02d}"

        if not wav_path.exists():
            logger.warning("WAV file not found (datalad get?): %s", wav_path)
            continue

        # Compute n_trs from audio duration
        try:
            with wave.open(str(wav_path), "rb") as wf:
                duration_s = wf.getnframes() / wf.getframerate()
        except Exception as exc:
            logger.warning("Cannot read WAV header for %s: %s", wav_name, exc)
            continue

        # Read pre-stimulus silence onset from events.tsv (single-row BIDS file)
        events_path = events_dir / f"task-{task_prefix}_run-{section:02d}_events.tsv"
        if not events_path.exists():
            logger.warning("events.tsv not found: %s (falling back to 4.0 s)", events_path)
            onset_s = 4.0
        else:
            ev = pd.read_csv(events_path, sep="\t")
            onset_s = float(ev.iloc[0]["onset"])

        onset_trs = int(onset_s / TR_DURATION)
        audio_trs = int(np.ceil(duration_s / TR_DURATION))
        n_trs = onset_trs + audio_trs

        runs.append({
            "run_id": run_id,
            "audio_path": str(wav_path),
            "n_trs": n_trs,
            "audio_trs": audio_trs,
            "onset_trs": onset_trs,
            "onset_s": onset_s,
            "section": section,
            "transcript_path": (
                str(transcript_path) if transcript_path and transcript_path.exists() else None
            ),
        })
        logger.info("  %s: %.1f s → %d audio TRs + %d onset TRs = %d total (onset=%.2f s)",
                     run_id, duration_s, audio_trs, onset_trs, n_trs, onset_s)

    return runs


def discover_runs(stimulus, config):
    """Discover runs for a given stimulus."""
    if stimulus == "friends":
        return discover_runs_friends(config)
    elif stimulus == "movie10":
        return discover_runs_movie10(config)
    elif stimulus == "harrypotter":
        return discover_runs_hp(config)
    elif stimulus.startswith("petitprince"):
        return discover_runs_pp(config)
    else:
        raise ValueError(f"Unknown stimulus: {stimulus}")


# ── Feature extraction dispatch ───────────────────────────────────────────


def process_run_audio(run_info, model, processor, out_dir, device, tmp_dir):
    """Extract audio features for one run."""
    video_path = run_info.get("video_path")
    audio_path = run_info.get("audio_path")
    run_id = run_info["run_id"]
    n_trs = run_info["n_trs"]

    # Extract audio from video if no direct audio file
    if audio_path is None and video_path is not None:
        audio_path = os.path.join(tmp_dir, f"{run_id}.wav")
        if not os.path.exists(audio_path):
            extract_audio_from_video(video_path, audio_path)
    elif audio_path is None:
        logger.warning("No audio source for run %s — skipping", run_id)
        return False

    # For PP: extract features for audio portion only, then prepend onset TRs
    onset_trs = run_info.get("onset_trs", 0)
    audio_trs = run_info.get("audio_trs", n_trs)

    layer_features = extract_audio_features(
        audio_path, model, processor, audio_trs, device=device,
    )

    # Zero-pad onset TRs so features are scan-aligned (mirrors HP text
    # where early TRs have empty text → zero-like embeddings)
    if onset_trs > 0:
        for layer_idx, features in layer_features.items():
            pad = np.zeros((onset_trs, features.shape[1]), dtype=features.dtype)
            layer_features[layer_idx] = np.concatenate([pad, features], axis=0)

    for layer_idx, features in layer_features.items():
        save_layer_features(out_dir, layer_idx, run_id, features)

    return True


def process_run_video(run_info, model, processor, out_dir, device):
    """Extract video features for one run."""
    video_path = run_info.get("video_path")
    run_id = run_info["run_id"]
    n_trs = run_info["n_trs"]

    if video_path is None:
        logger.warning("No video for run %s — skipping", run_id)
        return False

    layer_features = extract_video_features(
        video_path, model, processor, n_trs, device=device,
    )

    for layer_idx, features in layer_features.items():
        save_layer_features(out_dir, layer_idx, run_id, features)

    return True


def process_run_text(run_info, model, tokenizer, out_dir, device, config, window_trs):
    """Extract text features for one run."""
    run_id = run_info["run_id"]
    n_trs = run_info["n_trs"]

    # Load transcript based on type
    transcript_type = config["transcript_type"]

    if transcript_type == "tr_aligned_tsv":
        transcript_path = run_info.get("transcript_path")
        if transcript_path is None:
            logger.warning("No transcript for run %s — skipping", run_id)
            return False
        cumulative_text = load_transcript_friends_m10(transcript_path, n_trs)

    elif transcript_type == "hp_events_tsv":
        events_path = run_info["events_path"]
        cumulative_text = load_transcript_hp(events_path, n_trs)

    elif transcript_type == "pp_transcript":
        transcript_path = run_info.get("transcript_path")
        if transcript_path is None:
            logger.warning("No transcript for run %s — skipping", run_id)
            return False
        cumulative_text = load_transcript_pp(
            transcript_path,
            section=run_info["section"],
            n_trs=n_trs,
            onset_s=run_info["onset_s"],
        )

    else:
        raise ValueError(f"Unknown transcript type: {transcript_type}")

    layer_features = extract_text_features(
        cumulative_text, model, tokenizer, n_trs,
        window_trs=window_trs, device=device,
    )

    for layer_idx, features in layer_features.items():
        save_layer_features(out_dir, layer_idx, run_id, features)

    return True


# ── Main ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract layer-wise transformer features per run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audio features for Friends
  python script/08c_transformer_features.py --stimulus friends --model w2v-bert-2.0

  # Video features for Movie10
  python script/08c_transformer_features.py --stimulus movie10 --model dinov2-large

  # Text features for HP (default window_trs=4)
  python script/08c_transformer_features.py --stimulus harrypotter --model llama-3.2-3b

  # LLaMA W-sweep variant — writes to 08c_transformer_features_sweep_w3/...
  python script/08c_transformer_features.py --stimulus friends --model llama-3.2-3b \\
      --window_trs 3 --output_subdir_suffix _sweep_w3

  # Single run (debugging)
  python script/08c_transformer_features.py --stimulus friends --model llama-3.2-3b --run_id s01e01a
        """,
    )
    parser.add_argument(
        "--stimulus",
        type=str,
        default=None,
        choices=list(STIMULUS_CONFIG.keys()),
        help="Stimulus dataset (required unless --download_only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Transformer model to extract features from. "
             "Use 'all' with --download_only to cache all models.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Process a single run (for debugging).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--audio_cache_dir",
        type=str,
        default=None,
        help="Directory for cached extracted audio WAV files. "
             "If None, uses a temporary directory (cleaned up after).",
    )
    parser.add_argument(
        "--window_trs",
        type=int,
        default=4,
        help="Local-window span in TRs for LLaMA readout (default: 4). "
             "Per 2026-05-01_08c_llama_local_window_design.md §2.4 sweep grid: "
             "{1, 3, 6, 9}. Ignored for audio/video models.",
    )
    parser.add_argument(
        "--output_subdir_suffix",
        type=str,
        default="",
        help="Suffix appended to the '08c_transformer_features' output directory "
             "(e.g. '_sweep_w3' writes to '08c_transformer_features_sweep_w3/...'). "
             "Default empty (production path).",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download model weights to HF cache and exit (no extraction).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Download-only mode: cache model(s) and exit ────────────────────
    if args.download_only:
        models_to_download = (
            list(MODEL_REGISTRY.keys()) if args.model == "all"
            else [args.model]
        )
        for mk in models_to_download:
            logger.info("Downloading %s (%s) ...", mk, MODEL_REGISTRY[mk]["hf_id"])
            load_model(mk, device=args.device)
            logger.info("  %s cached successfully", mk)
        logger.info("All models cached — exiting (download-only mode)")
        sys.exit(0)

    if args.stimulus is None:
        logger.error("--stimulus is required (unless using --download_only)")
        sys.exit(1)

    stimulus = args.stimulus
    model_key = args.model
    config = STIMULUS_CONFIG[stimulus]

    # Validate model-stimulus compatibility
    required_modality = MODEL_MODALITY[model_key]
    if required_modality not in config["modalities"]:
        logger.error(
            "Model %s requires modality '%s' but stimulus '%s' only has %s",
            model_key, required_modality, stimulus, config["modalities"],
        )
        sys.exit(1)

    # ── Validate text-only flags ──────────────────────────────────────
    # Per 2026-05-01_08c_llama_local_window_design.md §3.1, --window_trs is
    # consumed only by the LLaMA local-window readout. We tolerate the
    # default value with non-text models (harmless) but reject an explicit
    # non-default to surface miswired SLURM exports.
    if required_modality != "text" and args.window_trs != 4:
        logger.error(
            "--window_trs %d was passed with --model %s (modality=%s); "
            "the local-window readout is text-only.",
            args.window_trs, model_key, required_modality,
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("08c - Transformer Feature Extraction")
    logger.info("=" * 60)
    logger.info("Stimulus: %s", stimulus)
    logger.info("Model: %s (%s)", model_key, MODEL_REGISTRY[model_key]["hf_id"])
    logger.info("Device: %s", args.device)
    if required_modality == "text":
        logger.info("Window TRs (LLaMA local pool): %d", args.window_trs)
    if args.output_subdir_suffix:
        logger.info("Output subdir suffix: %s", args.output_subdir_suffix)

    # ── Discover runs ─────────────────────────────────────────────────
    runs = discover_runs(stimulus, config)
    logger.info("Discovered %d runs", len(runs))

    if not runs:
        logger.warning("No runs found — exiting")
        sys.exit(0)

    # Filter to single run if requested
    if args.run_id:
        runs = [r for r in runs if r["run_id"] == args.run_id]
        if not runs:
            logger.error("Run %s not found", args.run_id)
            sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = os.path.join(
        SCRATCH_DIR, "output",
        f"08c_transformer_features{args.output_subdir_suffix}",
        config["output_subdir"], model_key,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    model, processor = load_model(model_key, device=args.device)

    # ── Extract features ──────────────────────────────────────────────
    n_success = 0
    n_skipped = 0
    n_trs_per_run = {}

    # Audio cache directory
    if required_modality == "audio":
        if args.audio_cache_dir:
            tmp_dir = args.audio_cache_dir
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_ctx = None
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            tmp_dir = tmp_ctx.name
    else:
        tmp_dir = None
        tmp_ctx = None

    try:
        for i, run_info in enumerate(runs):
            run_id = run_info["run_id"]
            n_trs = run_info["n_trs"]
            logger.info(
                "[%d/%d] Processing run %s (%d TRs)",
                i + 1, len(runs), run_id, n_trs,
            )

            # Check if already processed (all layers exist)
            n_layers = MODEL_REGISTRY[model_key]["n_layers"]
            first_layer = os.path.join(out_dir, "layer_00", f"{run_id}_raw.npy")
            last_layer = os.path.join(
                out_dir, f"layer_{n_layers - 1:02d}", f"{run_id}_raw.npy"
            )
            if os.path.exists(first_layer) and os.path.exists(last_layer):
                logger.info("  Already processed — skipping")
                # Record the actual on-disk TR count so metadata stays in sync
                # with the saved tensors regardless of which convention wrote them.
                actual_n_trs = int(np.load(first_layer, mmap_mode="r").shape[0])
                n_trs_per_run[run_id] = actual_n_trs
                n_success += 1
                continue

            try:
                if required_modality == "audio":
                    ok = process_run_audio(
                        run_info, model, processor, out_dir, args.device, tmp_dir
                    )
                elif required_modality == "video":
                    ok = process_run_video(
                        run_info, model, processor, out_dir, args.device
                    )
                elif required_modality == "text":
                    ok = process_run_text(
                        run_info, model, processor, out_dir, args.device, config,
                        window_trs=args.window_trs,
                    )
                else:
                    raise ValueError(f"Unknown modality: {required_modality}")

                if ok:
                    n_success += 1
                    n_trs_per_run[run_id] = n_trs
                else:
                    n_skipped += 1

            except Exception:
                logger.exception("  Error processing run %s", run_id)
                n_skipped += 1

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    # ── Save metadata ─────────────────────────────────────────────────
    extra_metadata = {}
    if required_modality == "text":
        extra_metadata.update(
            pooling="local_window",
            window_trs=args.window_trs,
            design_doc="the design notes",
        )
    if args.output_subdir_suffix:
        extra_metadata["output_subdir_suffix"] = args.output_subdir_suffix

    save_extraction_metadata(
        out_dir,
        model_key,
        stimulus,
        n_runs=n_success,
        n_trs_per_run=n_trs_per_run,
        run_ids=sorted(n_trs_per_run.keys()),
        **extra_metadata,
    )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Stimulus: %s, Model: %s", stimulus, model_key)
    logger.info("Processed: %d/%d runs", n_success, n_success + n_skipped)
    logger.info("Skipped: %d", n_skipped)
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
