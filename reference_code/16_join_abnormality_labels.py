#!/usr/bin/env python3
"""
16_join_abnormality_labels.py

STEP 2 of the temporal-labeling pipeline (HYBRID approach).

Joins CT-RATE's structured 18-way multi-abnormality labels onto the prior->current
pair manifest so that PRESENCE-change (new / resolved / present-both) is derived
DETERMINISTICALLY from data, instead of being guessed by the LLM. MedGemma is then
only responsible for SEVERITY direction (worse/stable/improved) on findings that are
present in BOTH studies, plus dynamic/static sentence extraction.

Presence-change rule per finding (prior_bit -> curr_bit):
    0 -> 1 : new
    1 -> 0 : resolved
    1 -> 1 : present_both   (severity decided later by the LLM)
    0 -> 0 : absent_both    (skipped; nothing there)

Inputs
------
data/ctrate/ctrate_pairs_enriched.csv                 (from 12_ctrate_enrich.py)
HF: ibrahimhamamci/CT-RATE
    dataset/multi_abnormality_labels/train_predicted_labels.csv
    dataset/multi_abnormality_labels/valid_predicted_labels.csv

Output
------
data/ctrate/ctrate_pairs_enriched_v2.csv   <- feed THIS to the MedGemma labeler
    adds columns:
      prior_labels        JSON {finding: 0/1}
      curr_labels         JSON {finding: 0/1}
      presence_changes    JSON {finding: "new"|"resolved"|"present_both"}   (absent_both omitted)
      new_findings        "; "-joined list
      resolved_findings   "; "-joined list
      present_both_findings "; "-joined list  (what the LLM must rate for severity)
      labels_found        1 if BOTH volumes were present in the label table, else 0
"""
import csv
import json
import os
import sys
from collections import Counter

csv.field_size_limit(10 ** 9)

REPO_ID = "ibrahimhamamci/CT-RATE"
OUT_DIR = "data/ctrate"
CACHE = os.path.join(OUT_DIR, "_labels")
IN_CSV = os.path.join(OUT_DIR, "ctrate_pairs_enriched.csv")
OUT_CSV = os.path.join(OUT_DIR, "ctrate_pairs_enriched_v2.csv")

LABEL_CSVS = [
    "dataset/multi_abnormality_labels/train_predicted_labels.csv",
    "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
]

# The 18 official CT-RATE abnormality labels (column order in the label CSVs).
FINDINGS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]


def resolve(path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, path, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"), local_dir=CACHE)


def load_labels():
    """VolumeName -> {finding: 0/1} for all 18 findings."""
    vec = {}
    for p in LABEL_CSVS:
        fp = resolve(p)
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            rd = csv.DictReader(f)
            missing = [c for c in FINDINGS if c not in rd.fieldnames]
            if missing:
                sys.exit(f"label CSV {p} missing expected columns: {missing}")
            for row in rd:
                vol = (row.get("VolumeName") or "").strip()
                if not vol:
                    continue
                vec[vol] = {c: (1 if str(row.get(c, "0")).strip() in ("1", "1.0") else 0)
                            for c in FINDINGS}
    return vec


def presence_change(pbit, cbit):
    if pbit == 0 and cbit == 1:
        return "new"
    if pbit == 1 and cbit == 0:
        return "resolved"
    if pbit == 1 and cbit == 1:
        return "present_both"
    return "absent_both"


def main():
    if not os.path.exists(IN_CSV):
        sys.exit(f"missing {IN_CSV} -- run scripts/12_ctrate_enrich.py first")
    print("loading structured 18-label table (train+valid, cached)...")
    vec = load_labels()
    print(f"  labeled volumes: {len(vec):,}")

    with open(IN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  pairs in manifest: {len(rows):,}")

    out_fields = list(rows[0].keys()) + [
        "prior_labels", "curr_labels", "presence_changes",
        "new_findings", "resolved_findings", "present_both_findings", "labels_found",
    ]

    n_found = 0
    tot = Counter()          # new/resolved/present_both/absent_both across all findings
    per_pair_changed = []    # number of new+resolved+present_both per pair
    for r in rows:
        pv = vec.get(r["prior_volume"])
        cv = vec.get(r["curr_volume"])
        if pv is None or cv is None:
            r["prior_labels"] = "" if pv is None else json.dumps(pv)
            r["curr_labels"] = "" if cv is None else json.dumps(cv)
            r["presence_changes"] = ""
            r["new_findings"] = r["resolved_findings"] = r["present_both_findings"] = ""
            r["labels_found"] = 0
            continue
        n_found += 1
        changes = {}
        new_f, res_f, both_f = [], [], []
        for c in FINDINGS:
            st = presence_change(pv[c], cv[c])
            tot[st] += 1
            if st == "absent_both":
                continue
            changes[c] = st
            if st == "new":
                new_f.append(c)
            elif st == "resolved":
                res_f.append(c)
            else:
                both_f.append(c)
        per_pair_changed.append(len(changes))
        r["prior_labels"] = json.dumps(pv)
        r["curr_labels"] = json.dumps(cv)
        r["presence_changes"] = json.dumps(changes)
        r["new_findings"] = "; ".join(new_f)
        r["resolved_findings"] = "; ".join(res_f)
        r["present_both_findings"] = "; ".join(both_f)
        r["labels_found"] = 1

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    print(f"\npairs with BOTH volumes labeled: {n_found:,}/{len(rows):,} "
          f"({100*n_found/max(len(rows),1):.1f}%)")
    print("\nstructured presence-change totals (summed over 18 findings x all pairs):")
    for k in ["new", "resolved", "present_both", "absent_both"]:
        print(f"  {k:>13}: {tot[k]:,}")
    if per_pair_changed:
        avg = sum(per_pair_changed) / len(per_pair_changed)
        both_avg = tot['present_both'] / max(n_found, 1)
        print(f"\n  avg changed findings/pair (new+resolved+both): {avg:.2f}")
        print(f"  avg present-both findings/pair (LLM will rate):  {both_avg:.2f}")
        print(f"  -> compare 'new' here to MedGemma's earlier new=101/50-pilot "
              f"(~{tot['new']/max(n_found,1):.2f}/pair) to gauge the precision gain")
    print(f"\nwrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
