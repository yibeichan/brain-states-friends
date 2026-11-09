#!/usr/bin/env python3
"""p7_side_root.py - lay out and check a side output root for a downstream rerun.

A side root is <root>/<name>/output/. Stages the rerun only reads are
symlinks to the published dataset's top-level stage directories; stages it
regenerates are real directories, so nothing a script creates (os.makedirs,
np.save) can land inside the dataset.

    linked (read only):   00_get_scan, 02_parcel_ts_avg, 03a_pca4combined_hmm,
                          and m10_03_projected when --link_m10_03
    real (regenerated):   04_combined_hdphmm, 05a_recurrence_analysis,
                          05e_temporal_trend_a1..a4, 05f_state_fc,
                          06a_state_temp_dynamics, 06b_transition_structure,
                          06e_transition_modules, m10_04_decoded,
                          m10_05_cross_validation, and m10_03_projected
                          unless linked

05e_temporal_trend_a4 parses --vt as a float, so for vt "0.90" it reads and
writes ".../vt0.9" where every other script uses ".../vt0.90". For such a vt
the per-subject directories get a local alias: vt0.9 -> vt0.90 in the stages
a4 reads, and vt0.90 -> vt0.9 in a4's own output. Both ends are inside the
side root.

--published_final links each file of the published final/vt{vt}/ into a real
final/vt{vt}/ directory (the gate run on the published model).

`check` resolves every regenerated per-subject path and fails if any lands
inside the published dataset.
"""
import argparse
import os
import sys

PARC = "atlas-4S156Parcels"
LINKED = ("00_get_scan", "02_parcel_ts_avg", "03a_pca4combined_hmm")
REGEN = ("04_combined_hdphmm", "05a_recurrence_analysis", "05e_temporal_trend_a1",
         "05e_temporal_trend_a2", "05e_temporal_trend_a3", "05e_temporal_trend_a4",
         "05f_state_fc", "06a_state_temp_dynamics", "06b_transition_structure",
         "06e_transition_modules", "m10_04_decoded", "m10_05_cross_validation")
A4_INPUTS = ("05a_recurrence_analysis", "05e_temporal_trend_a1", "05e_temporal_trend_a2",
             "05e_temporal_trend_a3")
FINAL_FILES = ("best_model.pkl", "decoded_states.pkl", "final_results.json",
               "pca_model.pkl", "state_covars.npy", "state_covars_parcel.npy",
               "state_means_parcel.npy", "state_means_pca.npy")


def inside(path, protected):
    real, prot = os.path.realpath(path), os.path.realpath(protected)
    return real == prot or real.startswith(prot.rstrip("/") + "/")


def sub_vt_dirs(out, stage, sub, vt):
    """Directories a stage reads or writes for one subject at one vt."""
    base = os.path.join(out, stage, PARC, sub)
    if stage == "04_combined_hdphmm":
        base = os.path.join(base, "final")
    return base, os.path.join(base, f"vt{vt}")


def float_vt_alias(vt):
    """The directory name 05e_a4 forms from vt, or None if it matches."""
    alias = f"vt{float(vt)}"
    return None if alias == f"vt{vt}" else alias


def _mkdir_real(path, published_out):
    os.makedirs(path, exist_ok=True)
    if os.path.islink(path) or inside(path, published_out):
        raise RuntimeError(f"{path} is a link or resolves into the dataset")


def setup(root, name, vt, subs, published_out, link_m10_03, published_final):
    out = os.path.join(root, name, "output")
    os.makedirs(out, exist_ok=True)
    if inside(out, published_out):
        raise RuntimeError(f"{out} resolves into the dataset")
    linked = list(LINKED) + (["m10_03_projected"] if link_m10_03 else [])
    regen = list(REGEN) + ([] if link_m10_03 else ["m10_03_projected"])
    for stage in linked:
        dst = os.path.join(out, stage)
        src = os.path.join(published_out, stage)
        if not os.path.isdir(src):
            raise RuntimeError(f"missing published input {src}")
        if os.path.islink(dst):
            if os.readlink(dst) != src:
                raise RuntimeError(f"{dst} links elsewhere: {os.readlink(dst)}")
        elif os.path.exists(dst):
            raise RuntimeError(f"{dst} exists and is not a link")
        else:
            os.symlink(src, dst)
    alias = float_vt_alias(vt)
    for stage in regen:
        for sub in subs:
            base, vt_dir = sub_vt_dirs(out, stage, sub, vt)
            if alias and stage == "05e_temporal_trend_a4":
                continue          # handled below: real dir is the float-named one
            _mkdir_real(vt_dir, published_out)
    if alias:
        for sub in subs:
            for stage in ("04_combined_hdphmm",) + A4_INPUTS:
                base, _ = sub_vt_dirs(out, stage, sub, vt)
                link = os.path.join(base, alias)
                if not os.path.lexists(link):
                    os.symlink(f"vt{vt}", link)
            base, vt_dir = sub_vt_dirs(out, "05e_temporal_trend_a4", sub, vt)
            # a4 writes into the float-named dir; others read the string-named one
            if os.path.isdir(vt_dir) and not os.path.islink(vt_dir):
                raise RuntimeError(f"{vt_dir} is a real directory; expected a link to {alias}")
            _mkdir_real(os.path.join(base, alias), published_out)
            if not os.path.lexists(vt_dir):
                os.symlink(alias, vt_dir)
    if published_final:
        for sub in subs:
            _, vt_dir = sub_vt_dirs(out, "04_combined_hdphmm", sub, vt)
            src_dir = os.path.join(published_out, "04_combined_hdphmm", PARC, sub,
                                   "final", f"vt{vt}")
            for fn in FINAL_FILES:
                dst = os.path.join(vt_dir, fn)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(src_dir, fn), dst)
    check(root, name, vt, subs, published_out, link_m10_03)


def check(root, name, vt, subs, published_out, link_m10_03):
    out = os.path.join(root, name, "output")
    regen = list(REGEN) + ([] if link_m10_03 else ["m10_03_projected"])
    bad = []
    for stage in regen:
        top = os.path.join(out, stage)
        if os.path.islink(top) or inside(top, published_out):
            bad.append(top)
        for sub in subs:
            for p in sub_vt_dirs(out, stage, sub, vt):
                if inside(p, published_out):
                    bad.append(p)
            alias = float_vt_alias(vt)
            if alias:
                p = os.path.join(sub_vt_dirs(out, stage, sub, vt)[0], alias)
                if os.path.lexists(p) and inside(p, published_out):
                    bad.append(p)
    if bad:
        raise SystemExit("regenerated paths resolve into the dataset:\n  " + "\n  ".join(bad))
    print(f"ok: {out} ({len(regen)} regenerated stages x {len(subs)} subjects outside "
          f"{os.path.realpath(published_out)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["setup", "check"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--vt", required=True)
    ap.add_argument("--subs", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--published_out", required=True)
    ap.add_argument("--link_m10_03", action="store_true")
    ap.add_argument("--published_final", action="store_true")
    a = ap.parse_args()
    if a.action == "setup":
        setup(a.root, a.name, a.vt, a.subs, a.published_out, a.link_m10_03, a.published_final)
    else:
        check(a.root, a.name, a.vt, a.subs, a.published_out, a.link_m10_03)


if __name__ == "__main__":
    sys.exit(main())
