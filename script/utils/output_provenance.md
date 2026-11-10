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
   Outputs of those stages are then refused if their code lacks the fix. Scope the line to the stages whose code path the fix changes. Downstream stages read the fixed outputs, not the fixed code. The checks read the floor and `DUPLICATE_RUN_SESSIONS` from `origin/main` (`git show`), not from the checkout running them, so an old checkout cannot pass outputs that a newer floor line rules out. They fetch `origin/main` first; a failed fetch is a failure (`--no-fetch` skips the fetch on purpose).
3. **Reruns go to a side root, never in place.** A side root is a directory with its own `output/` tree. Run with `SCRATCH_DIR` pointing at it, and link unchanged inputs in as symlinks. Link files, not directories, for a stage that writes next to its inputs: 04rc writes into 04's fold directories, so it would write through a linked directory into the dataset. `promote.py` refuses a linked directory that has new or modified files in the dataset. Promote the results with `promote.py`.
4. **A promotion declares what changes.** Any recomputed file outside the declared change set must equal the dataset's copy.
5. **A promotion has controls that can fail.** At least one control is a recomputed file inside the change set, compared with a reference version. At least one control inside the change set, the positive control, must show the expected change: a numeric difference of at least `min_diff`, with shape, keys and NaN positions intact. A control on a unit the change cannot affect proves nothing.

## Provenance records

Stages that read BOLD runs write `provenance_<stage>.json` next to their outputs. These are currently 03a, 04rc, 05e a1, 05e a3 and 05f (`check_code.PROVENANCE_STAGES`). Each record holds:

| Field | Content |
|---|---|
| `code` | commit, branch, `dirty` (modified tracked files or untracked `.py` under `script/`, when the code was loaded or when the record was written; both file lists are kept), `head_changed` (HEAD moved during the run) |
| `scratch_dir`, `output_dataset_head` | the output root the stage read from and wrote to |
| `sessions_read` | the acquisition session read for every run resolved through `pick_run_session` |
| `duplicate_run_sessions_pinned` | `DUPLICATE_RUN_SESSIONS` for the participant at run time |
| `inputs` | every input file read through the loaders, with its size and md5. Annexed files are identified from their MD5E key without reading them, so the same bytes have the same identity whether annexed, linked or copied |
| `runs` | per-run `n_ts`, `n_vit`, `n_used`, and skipped runs with the reason |
| `outputs` | every file the record describes, relative to its directory, with size and md5: by default each file under the stage's output directory written since the process started; 04rc names its two files, because it writes into stage 04's fold directories |

A record is written only when the outputs are recomputed. A checkpoint skip leaves the old record in place. A record covers exactly the files it lists with those checksums. An output not listed in its nearest record (the closest `provenance_<stage>.json` in its directory or above, within the stage directory), or listed with another checksum, is not covered. That catches outputs left from an older run and files a crashed or partial rerun rewrote. A stage that reuses cached files without rewriting them leaves those files uncovered: rerun it with its force option.

To add a stage:
1. Load inputs through the shared loaders (or call `provenance.record_input`).
2. Call `provenance.write_provenance(out_dir, "<stage>", sub_id, parcellation)` after the outputs are written.
3. Add its output directory to `check_code.STAGE_TOPS`.

Stages 00 and 02, which define the BOLD content, do not yet write records. Their outputs are identified downstream through the annex keys in `inputs`.

## Saving with checks

```bash
bash script/utils/datalad_save.sh --stage 05f --subject sub-06
sbatch --export=STAGE=05f,SUBJECT_ID=sub-06 script/utils/datalad_save.sh
```

The code check runs on every save. It takes the save lock first, so it sees exactly what the save will commit. It checks the files the save would commit, the new or modified ones per `git status` under the save path; files already in the dataset are not checked again. It fails the save in any of these cases:
- `origin/main` cannot be fetched;
- a record is from a commit not on `origin/main`;
- a record's commit lacks a floor fix for its stage;
- a record says the checkout was dirty;
- a record read a duplicated run from a session other than the pinned one;
- a new or modified output of a provenance stage is not covered by its nearest record. This includes 04rc's files inside a stage-04 save.
- the save deletes an output that its record still lists.

Files of stages that write no provenance get a weaker check, and the output says so (`WEAK CHECK`). The save-time checkout must pass the floor and be clean: no modified tracked files and no untracked `.py` under `script/`.

- **Skipping the check.** Pass `--no-check-code` or set `CHECK_CODE=0`. Do this only for a save that cannot pass it, and give the reason with `--skip-reason "<text>"` (or `SKIP_REASON`). The commit message then ends with `code_check: skipped (<reason>)` and the save prints a warning.
- **GitHub unreachable.** `--no-fetch` (or `CHECK_CODE_NO_FETCH=1`) checks against the local `origin/main`, and the commit message says it was not fetched. A failed fetch is retried once, since parallel saves race for the ref lock.
- **Values.** `CHECK_CODE` accepts 1/true/yes and 0/false/no; any other value stops the save.
- **The commit message records the code floor from `origin/main`, the one the check enforced.**

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

`--side-root` is the rerun's root directory, the one that holds `output/` and receives the `promote/` plans and logs, not `output/` itself. A side root without `output/` is refused before anything is written.

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
- `allow_input_changes`: a table of input-key patterns to reasons, for example `allow_input_changes = {"parcel_ts:*" = "02 rerun"}`. Only matching inputs may differ from the dataset's record; inputs are compared by content. Sessions appear as `session:<run>`.
- `weak_stages`: a reason. Planned files of stages that write no provenance are then accepted on the manifest's `code_commit` alone, with a warning. Without it they block the promotion.
- `no_positive_control`: a reason; then no positive control is required.

Globs use `**` for any number of whole directories; `*` and `?` stay within one path segment. A change glob must start with a literal stage directory (`05f_state_fc/...`), so no glob can declare several stages at once.

How files are compared:
- `.npy`/`.npz`: shapes and dtype kinds exact; NaN and inf positions exact; finite values within `atol + rtol·|ref|` (defaults 1e-12 and 1e-9).
- `.json`: dict keys exact, list order kept; `bool` and strings exact; int vs float a structural difference; numbers within tolerance.
- A structural difference (shape, dtype, keys, NaN positions, strings, bytes) never counts as the expected change for a positive control.
- `.csv`/`.tsv`: header exact; numeric cells within tolerance, others exact.
- Other files: compared by bytes.
- Figures outside the change set: a warning (renders are not byte-stable); `--strict-figures` makes them a failure.

A reference version at a past dataset commit is read from git: plain files from the blob, annexed files through `git annex contentlocation`. That content must be present locally (`datalad get`).

A planned file that the dataset's `.gitignore` matches is refused, since it would be copied but never saved.

A record is promoted with every output it lists, and it lists every file the run wrote, figures included, whose bytes change on every render. Declare the record's whole directory in `change` (for example `05f_state_fc/atlas-4S156Parcels/sub-06/**`); naming only the files you meant to change fails. A listed file that equals the dataset's copy within tolerance but differs in bytes is copied too, so the dataset matches the record's checksums. Such a file outside the change set is refused. A dataset file that would remain under a promoted record without being listed in it is refused before anything is saved, inside or outside the change set, and even with `keep_unreproduced`. An untracked file in the dataset, as a crashed promotion leaves, is never taken as the dataset's version: inside the change set it is promoted as NEW, outside it the promotion stops.

`--execute` refuses to start when `origin/main` cannot be fetched. It then works under a lock (`.git/promote.lock`); if the dataset's HEAD moved after the checks ran, it stops without changing anything. Otherwise it does the following:
- Reruns every check.
- Copies each planned file through a temporary file whose md5 must equal the checksum taken when the file was checked; a file that changed since is not promoted.
- Runs one `datalad save`, without `--amend`, on the planned files' parent directories, moved up until there are at most 500. Before it runs, nothing but planned paths may be modified or untracked under those directories. `datalad_save.sh` waits up to 20 minutes for the same lock and holds it through its code check and save (its job limit is 60 minutes), releasing it before pushing, so the two never commit at once.
- Verifies the result:
  - the promotion is one commit directly on the starting HEAD, and the paths it changes are exactly the planned ones;
  - every path is tracked and clean;
  - the saved content equals the checked content;
  - every output under each promoted record's directory is covered by its record.

If any step fails, the log lists the NEW and UPDATE files and the starting HEAD for recovery. It does not push. The save runs `uv run --extra datalad datalad`; where git-annex comes from another environment, set `PROMOTE_DATALAD` to the datalad command, for example `PROMOTE_DATALAD="micromamba run -n <env> datalad"`. The commit message records the code commit, the code floor, the side root and the manifest's sha256. Plans and logs are written to `<side root>/promote/`.
