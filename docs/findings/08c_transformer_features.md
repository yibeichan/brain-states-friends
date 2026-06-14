# Findings: 08c Transformer Feature Extraction

_Script: `script/08c_transformer_features.py`. Tier: MAIN (R4b, Methods)._

_Extracts per-run, per-layer activations from three frozen pretrained transformers (audio, video, text) across five naturalistic stimuli at TR resolution; stimulus-level output (no sub_id)._

## Method (as run)

- **Models:** Wav2VecBert 2.0 (audio, 24 layers x 1024 dim), DINOv2-large (video, 24 layers x 1024 dim), LLaMA 3.2 3B (text, 28 layers x 3072 dim); all frozen, TRIBEv2-validated backbones
- **Stimuli:** friends (292 runs), movie10 (44 runs), harrypotter (7 runs, text-only), petitprince_fr (9 runs, audio+text), petitprince_en (9 runs, audio+text)
- **TR:** 1.49 s; features mean-pooled within each TR window
- **Aggregation:** audio: 30 s chunks, 5 s overlap, mean-pool per TR; video: mean-pool patch tokens (not CLS) per TR; text: mean-pool token activations per TR, cumulative context window
- **Output shape:** (n_trs, hidden_dim) float32 per layer per run; no PCA at this stage (PCA fitted lazily in 08d)
- **Run discovery:** friends: transcript-driven (TSV row count); movie10: events.tsv-driven, ceil((onset+duration)/TR); harrypotter: word-level events.tsv, ceil(last_word_onset/TR); petitprince: WAV duration + events.tsv onset_s, exact scan-time formula (onset_s + word.onset)/TR
- **Scope:** stimulus-level (one output per stimulus x model, shared across all subjects); not per-subject

## Results

### Bundle Inventory

| Stimulus | Model | n_runs | n_layers | hidden_dim | Files on disk | Status |
|---|---|---:|---:|---:|---:|---|
| friends | w2v-bert-2.0 | 292 | 24 | 1024 | 7008 | OK |
| friends | dinov2-large | 292 | 24 | 1024 | 7008 | OK |
| friends | llama-3.2-3b | 292 | 28 | 3072 | 8176 | OK |
| movie10 | w2v-bert-2.0 | 44 | 24 | 1024 | 1056 | OK |
| movie10 | dinov2-large | 44 | 24 | 1024 | 1056 | OK |
| movie10 | llama-3.2-3b | 44 | 28 | 3072 | 1232 | OK |
| harrypotter | llama-3.2-3b | 7 | 28 | 3072 | 196 | OK |
| petitprince_fr | w2v-bert-2.0 | 9 | 24 | 1024 | 216 | OK |
| petitprince_fr | llama-3.2-3b | 9 | 28 | 3072 | 252 | OK |
| petitprince_en | w2v-bert-2.0 | 9 | 24 | 1024 | 216 | OK |
| petitprince_en | llama-3.2-3b | 9 | 28 | 3072 | 252 | OK |

All 11 (stimulus x model) bundles complete; all layer directories present. File counts equal n_runs x n_layers for every bundle.

### Run-level TR Counts

| Stimulus | n_runs | min n_trs | max n_trs | mean n_trs |
|---|---:|---:|---:|---:|
| friends | 292 | 427 | 591 | 472 |
| movie10 | 44 | 373 | 496 | 407 |
| harrypotter | 7 | 373 | 566 | 480 |
| petitprince_fr | 9 | 392 | 506 | 437 |
| petitprince_en | 9 | 353 | 490 | 417 |

Note: movie10 has a +1 TR convention drift between 27 pre-2026-04-09 runs (bourne/wolf, transcript-row count) and 17 later runs (figures/life, ceil formula). The metadata.json for consolidated bundles records the actual on-disk shape, not the discovery-computed value.

### Modality Coverage by Stimulus

| Stimulus | Audio (w2v-bert-2.0) | Video (dinov2-large) | Text (llama-3.2-3b) |
|---|:---:|:---:|:---:|
| friends | yes | yes | yes |
| movie10 | yes | yes | yes |
| harrypotter | no | no | yes |
| petitprince_fr | yes | no | yes |
| petitprince_en | yes | no | yes |

Harry Potter is text-only (RSVP word-by-word, no audio/video stream). Petit Prince is audio+text only (no video).

## Outputs

- output/08c_transformer_features/{stimulus}/{model}/layer_00/{run_id}_raw.npy: (n_trs, hidden_dim) float32
- output/08c_transformer_features/{stimulus}/{model}/layer_NN/{run_id}_raw.npy: layers 00 through 23 (audio/video) or 27 (text)
- output/08c_transformer_features/{stimulus}/{model}/metadata.json: model info, run list, n_trs per run (llama metadata.json shows n_runs=1 due to skipped consolidation pass; actual file counts are correct)
- output/pp_annotations/EN/lppEN_word_information.csv: word-level annotations for PP-EN (downloaded from OpenNeuro ds003643)
- output/pp_annotations/FR/lppFR_word_information.csv: word-level annotations for PP-FR
