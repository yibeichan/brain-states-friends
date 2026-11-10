# Output provenance and promotion

The output dataset (`$SCRATCH_DIR/output`, a DataLad dataset) must hold only results that current code on `main` reproduces. Three tools enforce this:

- `script/utils/provenance.py` records what produced each output.
- `script/utils/check_code.py` checks the code against `main` and the code floor.
- `script/utils/promote.py` gates copying a rerun into the dataset.

## Rules

1. **Results come from `main`.** A result computed on a PR branch is not saved to the dataset. Merge the PR, then rerun the stage from `main`. The repository squash-merges, so a branch commit never becomes an ancestor of `main`; `check_code.py` therefore refuses branch commits even after their PR has merged.
2. **Correctness fixes raise the code floor.** When a merged fix changes results, the same PR adds a line to `script/utils/code_floor.txt`:
   ```
   <full sha> <stage keys, comma-separated, or *> <reason>
   ```
   Outputs of those stages are then refused if their code lacks the fix. Scope the line to the stages whose code path the fix changes. Downstream stages read the fixed outputs, not the fixed code.
3. **Reruns go to a side root, never in place.** A side root is a directory with its own `output/` tree. Run with `SCRATCH_DIR` pointing at it, and link unchanged inputs in as symlinks. Link files, not directories, for a stage that writes next to its inputs: 04rc writes into 04's fold directories, so it would write through a linked directory into the dataset. `promote.py` refuses a linked directory that has new or modified files in the dataset. Promote the results with `promote.py`.
4. **A promotion declares what changes.** Any recomputed file outside the declared change set must equal the dataset's copy.
5. **A promotion has controls that can fail.** At least one control is a recomputed file inside the change set, compared with a reference version. At least one control inside the change set, the positive control, must show the expected change: a numeric difference of at least `min_diff`, with shape, keys and NaN positions intact. A control on a unit the change cannot affect proves nothing.

## Provenance records

Stages that read BOLD runs write `provenance_<stage>.json` next to their outputs. These are currently 03a, 04rc, 05e a1, 05e a3 and 05f (`check_code.PROVENANCE_STAGES`). Each record holds:

| Field | Content |
|---|---|
| `code` | commit, branch, `dirty` (modified tracked files or untracked `.py` under `script/`), `head_changed` (HEAD moved during the run) |
| `scratch_dir`, `output_dataset_head` | the output root the stage read from and wrote to |
| `sessions_read` | the acquisition session read for every run resolved through `pick_run_session` |
| `duplicate_run_sessions_pinned` | `DUPLICATE_RUN_SESSIONS` for the participant at run time |
| `inputs` | every input file read through the loaders: annex key (from the symlink target, so no hashing) or sha256 |
| `runs` | per-run `n_ts`, `n_vit`, `n_used`, and skipped runs with the reason |
| `covers` | the files the record describes, when the directory also holds another stage's files (04rc writes into 04's fold directories) |

A record is written only when the outputs are recomputed. A checkpoint skip leaves the old record in place.

To add a stage:
1. Load inputs through the shared loaders (or call `provenance.record_input`).
2. Call `provenance.write_provenance(out_dir, "<stage>", sub_id, parcellation)` after the outputs are written.
3. Add the stage key to `check_code.PROVENANCE_STAGES` and its output directory to `promote._STAGE_TOPS`.

Stages 00 and 02, which define the BOLD content, do not yet write records. Their outputs are identified downstream through the annex keys in `inputs`.

## Saving with checks

```bash
bash script/utils/datalad_save.sh --stage 05f --subject sub-06 --check-code
sbatch --export=STAGE=05f,SUBJECT_ID=sub-06,CHECK_CODE=1 script/utils/datalad_save.sh
```

With `--check-code`, the save fails in any of these cases:
- a record under the save path is from a commit not on `origin/main`;
- a record's commit lacks a floor fix for its stage;
- a record says the checkout was dirty;
- a record read a duplicated run from a session other than the pinned one;
- the stage writes provenance and no record is present.

Every record under the save path is checked, whichever stage wrote it. A stage that writes provenance needs a record of its own stage in every subject directory: with `--subject all`, that means every `sub-*` directory. For other stages, the save-time checkout is checked as well, and the output says the check is weak (`WEAK CHECK`). `CHECK_CODE` accepts 1/true/yes and 0/false/no; any other value stops the save. The check is off by default while it is being adopted.

`check_code.py` also works on its own:

```bash
python script/utils/check_code.py commit <sha> --stage 05f
python script/utils/check_code.py tree "$SCRATCH_DIR/output/05f_state_fc/atlas-4S156Parcels/sub-06" --stage 05f
```

## Promoting a side root

```bash
python script/utils/promote.py --side-root "$SIDE" --manifest promote.toml          # dry run
python script/utils/promote.py --side-root "$SIDE" --manifest promote.toml --execute
```

Example manifest:

```toml
message = "sub-06 05f recomputed on code with the duplicate-run pin"
code_commit = "<sha the side root ran>"
change = [
  "05f_state_fc/atlas-4S156Parcels/sub-06/**",
]

[[control]]          # must match a known-good version
path = "05f_state_fc/atlas-4S156Parcels/sub-06/vt0.95/fc_similarity_corr_rv.npy"
reference = "<dataset commit of the known-good version>"
why = "05f loader path: the duplicated run must be read from its pinned session"

[[control]]          # positive control: the change must actually happen
path = "05f_state_fc/atlas-4S156Parcels/sub-06/vt0.95/fc_similarity_corr_rv.npy"
reference = "HEAD"
expect = "differ"
min_diff = 1e-4      # required; larger than atol
why = "replaces the version computed without the pin"
```

Optional keys:
- `scope` and `exclude`: side-root globs to consider and to leave out.
- `[tolerance]`: `rtol` and `atol`.
- `[[override]]`: per-file `glob` with `rtol`, `atol`, `decimals` or `ignore_keys`.
  - `decimals = d` is for values rounded to `d` places; it allows a change of one unit in the last place, `atol` 1.5·10⁻ᵈ.
  - `ignore_keys` lists JSON keys to skip, such as timestamps.
  - Controls on a matching file use the same settings.
- `keep_unreproduced`: a reason; dataset files in the change set that the side root does not produce are then kept with a warning. Without it they block the promotion.
- `allow_input_changes`: a reason; input logs may then differ from the dataset's records.
- `no_positive_control`: a reason; then no positive control is required.

Globs use `**` for any number of whole directories; `*` and `?` stay within one path segment. A bare `**` is refused as a change glob.

How files are compared:
- `.npy`/`.npz`: shapes and dtype kinds exact; NaN and inf positions exact; finite values within `atol + rtol·|ref|` (defaults 1e-12 and 1e-9).
- `.json`: dict keys exact, list order kept; `bool` and strings exact; int vs float a structural difference; numbers within tolerance.
- A structural difference (shape, dtype, keys, NaN positions, strings, bytes) never counts as the expected change for a positive control.
- `.csv`/`.tsv`: header exact; numeric cells within tolerance, others exact.
- Other files: compared by bytes.
- Figures outside the change set: a warning (renders are not byte-stable); `--strict-figures` makes them a failure.

A reference version at a past dataset commit is read from git: plain files from the blob, annexed files through `git annex contentlocation`. That content must be present locally (`datalad get`).

`--execute` reruns every check, copies the planned files, runs `datalad save` on exactly those paths (500 per call, amended into one commit), and verifies that the saved content equals the side root's. If a copy or save fails, the output says the dataset has unsaved copies and how to revert them. It does not push. The save runs `uv run --extra datalad datalad`; where git-annex comes from another environment, set `PROMOTE_DATALAD` to the datalad command, for example `PROMOTE_DATALAD="micromamba run -n <env> datalad"`. The commit message records the code commit, the code floor, the side root and the manifest's sha256. Plans and logs are written to `<side root>/promote/`.
