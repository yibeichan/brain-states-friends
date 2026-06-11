#!/usr/bin/env python
"""
Extract relative acquisition times for Friends functional runs from BIDS scans.tsv.

For each subject, this script scans:
    $DATA_DIR/cneuromod/friends/sub-*/ses-*/*_scans.tsv

It keeps only functional BOLD runs and writes one CSV per subject with:
    - run_id
    - session_id
    - rel_acq_time
    - filename

Output:
    {SCRATCH_DIR}/output/00_get_scan/{sub_id}/{sub_id}_run_acquisition_times.csv

Notes
-----
- `rel_acq_time` is copied verbatim from each session's `_scans.tsv`; this script
  does not recompute or normalize time across sessions.
- Output remains file-level metadata (one row per BOLD file), not episode-level
  aggregation.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


FUNC_BOLD_RE = re.compile(
    r"^func/(?P<basename>sub-[^_]+_ses-[^_]+_task-(?P<run_id>[^_]+)_bold\.nii\.gz)$"
)
SESSION_RE = re.compile(r"(ses-\d+)")
SUBJECT_RE = re.compile(r"(sub-\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract relative acquisition time per Friends functional run."
    )
    parser.add_argument(
        "--sub_id",
        type=str,
        default=None,
        help='Subject ID to process (e.g., "sub-01"). If omitted, process all subjects.',
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to Friends BIDS root containing sub-*/ses-*/*_scans.tsv. "
             "Defaults to $DATA_DIR/cneuromod/friends.",
    )
    return parser.parse_args()


def extract_subject_id(scans_path: Path) -> str:
    match = SUBJECT_RE.search(scans_path.as_posix())
    if not match:
        raise ValueError(f"Could not parse subject ID from path: {scans_path}")
    return match.group(1)


def extract_session_id(scans_path: Path) -> str:
    match = SESSION_RE.search(scans_path.as_posix())
    if not match:
        raise ValueError(f"Could not parse session ID from path: {scans_path}")
    return match.group(1)


def parse_run_record(row, session_id):
    filename = (row.get("filename") or "").strip()
    match = FUNC_BOLD_RE.match(filename)
    if not match:
        return None

    rel_acq_time = row.get("rel_acq_time")
    if rel_acq_time is None or str(rel_acq_time).strip() == "":
        raise ValueError(
            f"Missing rel_acq_time for functional run {filename} in {session_id}"
        )

    return {
        "run_id": match.group("run_id"),
        "session_id": session_id,
        "rel_acq_time": str(rel_acq_time).strip(),
        "filename": filename,
    }


def load_subject_rows(scans_paths):
    rows = []

    for scans_path in sorted(scans_paths):
        session_id = extract_session_id(scans_path)

        with scans_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {"filename", "rel_acq_time"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing required columns [{missing_str}] in {scans_path}"
                )

            for row in reader:
                run_record = parse_run_record(row, session_id)
                if run_record is not None:
                    rows.append(run_record)

    rows.sort(key=lambda row: (row["session_id"], float(row["rel_acq_time"]), row["run_id"]))
    return rows


def write_subject_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "session_id", "rel_acq_time", "filename"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        print("ERROR: SCRATCH_DIR must be set in the .env file", file=sys.stderr)
        sys.exit(1)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            print("ERROR: DATA_DIR must be set in the .env file (or pass --dataset_root)", file=sys.stderr)
            sys.exit(1)
        dataset_root = Path(data_dir) / "cneuromod" / "friends"
    if not dataset_root.exists():
        print(f"ERROR: Dataset root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    if args.sub_id is None:
        scans_paths = sorted(dataset_root.glob("sub-*/ses-*/*_scans.tsv"))
    else:
        scans_paths = sorted(dataset_root.glob(f"{args.sub_id}/ses-*/*_scans.tsv"))

    if not scans_paths:
        target = args.sub_id if args.sub_id else "all subjects"
        print(f"ERROR: No scans.tsv files found for {target} in {dataset_root}", file=sys.stderr)
        sys.exit(1)

    subject_to_paths = {}
    for scans_path in scans_paths:
        subject_to_paths.setdefault(extract_subject_id(scans_path), []).append(scans_path)

    output_root = Path(scratch_dir) / "output" / "00_get_scan"

    for sub_id in sorted(subject_to_paths):
        rows = load_subject_rows(subject_to_paths[sub_id])
        out_path = output_root / sub_id / f"{sub_id}_run_acquisition_times.csv"
        write_subject_csv(rows, out_path)
        print(f"Wrote {len(rows)} runs for {sub_id}: {out_path}")


if __name__ == "__main__":
    main()
