#!/usr/bin/env python3
"""
12_ctrate_enrich.py

STEP 1 of the temporal-labeling pipeline: build a clean, date-ordered prior->current
pair manifest from CT-RATE report + metadata CSVs. Metadata only (no volumes).

What it does
------------
1. Loads the report CSVs (VolumeName, ClinicalInformation_EN, Technique_EN,
   Findings_EN, Impressions_EN) and the DICOM metadata CSVs (StudyDate, PatientAge,
   Manufacturer, ManufacturerModelName, ConvolutionKernel, spacing, ...).
2. Groups volumes into STUDIES per patient (split_patientID + scanID letter),
   picking one representative reconstruction per study.
3. Orders each patient's studies by REAL StudyDate (not the a<b<c guess), forms
   consecutive prior->current pairs, and computes delta_days.
4. Drops degenerate pairs (missing date, same-day) so silver labels aren't polluted.
5. Emits clean report text = Findings_EN + Impressions_EN only (drops boilerplate
   Technique_EN), plus scanner-match flags and a comparison-language flag measured
   on the clean text.

Output
------
data/ctrate/ctrate_pairs_enriched.csv   <- feed THIS to the MedGemma labeler
data/ctrate/ctrate_enrich_summary.txt
prints a delta_days distribution.
"""
import csv
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime

REPO_ID = "ibrahimhamamci/CT-RATE"
OUT_DIR = "data/ctrate"
CACHE = os.path.join(OUT_DIR, "_reports")

REPORT_CSVS = [
    "dataset/radiology_text_reports/train_reports.csv",
    "dataset/radiology_text_reports/validation_reports.csv",
]
META_CSVS = [
    "dataset/metadata/train_metadata.csv",
    "dataset/metadata/validation_metadata.csv",
]

import re
COMPARISON_RE = re.compile(
    "|".join([r"increas", r"decreas", r"enlarg", r"larger", r"smaller", r"interval",
              r"compar", r"prior", r"previous", r"unchanged", r"stable", r"progress",
              r"regress", r"worsen", r"improv", r"resolv", r"redemonstrat",
              r"\bnew\b", r"since the"]),
    re.IGNORECASE,
)


def resolve(path):
    """Return local cached path for an HF file (downloads if missing)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, path, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"), local_dir=CACHE)


def load_reports():
    reports = {}
    for p in REPORT_CSVS:
        fp = resolve(p)
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                vol = (row.get("VolumeName") or "").strip()
                if not vol:
                    continue
                reports[vol] = {
                    "clinical": (row.get("ClinicalInformation_EN") or "").strip(),
                    "findings": (row.get("Findings_EN") or "").strip(),
                    "impression": (row.get("Impressions_EN") or "").strip(),
                }
    return reports


def load_meta():
    meta = {}
    keep = ["StudyDate", "PatientAge", "PatientSex", "Manufacturer",
            "ManufacturerModelName", "ConvolutionKernel", "SeriesDescription",
            "XYSpacing", "ZSpacing", "Rows", "Columns", "NumberofSlices"]
    for p in META_CSVS:
        fp = resolve(p)
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                vol = (row.get("VolumeName") or "").strip()
                if not vol:
                    continue
                meta[vol] = {k: (row.get(k) or "").strip() for k in keep}
    return meta


def parse_vol(vol):
    base = vol.replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("_")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]  # split, pid, scan, recon


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:10] if "-" in s else s[:8], fmt)
        except Exception:
            continue
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading report + metadata CSVs (cached)...")
    reports = load_reports()
    meta = load_meta()
    print(f"  reports: {len(reports):,}   metadata rows: {len(meta):,}")

    # ---- group into studies: (patient, scanID) -> representative volume + date ----
    # a study = one scanID letter for a patient; pick the smallest reconstruction that
    # actually has a StudyDate; fall back to smallest recon overall.
    studies = defaultdict(dict)  # patient -> {scanID: {"vol":..., "date":dt, ...}}
    for vol in reports:
        pv = parse_vol(vol)
        if not pv:
            continue
        split, pid, scan, recon = pv
        patient = f"{split}_{pid}"
        m = meta.get(vol, {})
        dt = parse_date(m.get("StudyDate"))
        cur = studies[patient].get(scan)
        cand = {"vol": vol, "date": dt, "meta": m}
        if cur is None:
            studies[patient][scan] = cand
        else:
            # prefer a representative that HAS a date, then smallest volume name
            cur_has = cur["date"] is not None
            cand_has = dt is not None
            if (cand_has and not cur_has) or (cand_has == cur_has and vol < cur["vol"]):
                studies[patient][scan] = cand

    multi = {p: s for p, s in studies.items() if len(s) >= 2}
    print(f"  patients with >=2 studies: {len(multi):,}")

    # ---- build date-ordered consecutive pairs ----
    pairs = []
    dropped_nodate = 0
    dropped_sameday = 0
    for patient, scanmap in multi.items():
        items = list(scanmap.values())
        # order by (date, scanID); undated sort last but we drop them below
        def sort_key(it):
            v = parse_vol(it["vol"])
            scan = v[2] if v else "z"
            return (it["date"] or datetime.max, scan)
        items.sort(key=sort_key)
        for i in range(len(items) - 1):
            prior, curr = items[i], items[i + 1]
            pd_, cd_ = prior["date"], curr["date"]
            if pd_ is None or cd_ is None:
                dropped_nodate += 1
                continue
            delta = (cd_ - pd_).days
            if delta <= 0:
                dropped_sameday += 1
                continue
            pr = reports[prior["vol"]]
            cr = reports[curr["vol"]]
            pm, cm = prior["meta"], curr["meta"]
            curr_clean = (cr["findings"] + " " + cr["impression"]).strip()
            prior_clean = (pr["findings"] + " " + pr["impression"]).strip()
            pairs.append({
                "patient": patient,
                "prior_volume": prior["vol"],
                "curr_volume": curr["vol"],
                "prior_date": pd_.strftime("%Y-%m-%d"),
                "curr_date": cd_.strftime("%Y-%m-%d"),
                "delta_days": delta,
                "patient_age": cm.get("PatientAge", ""),
                "patient_sex": cm.get("PatientSex", ""),
                "manufacturer_match": int(pm.get("Manufacturer", "") == cm.get("Manufacturer", "") and cm.get("Manufacturer", "") != ""),
                "model_match": int(pm.get("ManufacturerModelName", "") == cm.get("ManufacturerModelName", "")),
                "kernel_match": int(pm.get("ConvolutionKernel", "") == cm.get("ConvolutionKernel", "")),
                "curr_has_comparison": int(bool(COMPARISON_RE.search(curr_clean))),
                "prior_findings": pr["findings"],
                "prior_impression": pr["impression"],
                "curr_findings": cr["findings"],
                "curr_impression": cr["impression"],
                "curr_clinical": cr["clinical"],
            })

    print(f"  built {len(pairs):,} dated pairs "
          f"(dropped {dropped_sameday:,} same-day, {dropped_nodate:,} missing-date)")

    # ---- comparison language on CLEAN text (Findings+Impressions only) ----
    if pairs:
        cmp_frac = sum(p["curr_has_comparison"] for p in pairs) / len(pairs)
        print(f"  comparison language in CLEAN current text: {100*cmp_frac:.1f}%")

    # ---- delta_days distribution ----
    buckets = Counter()
    for p in pairs:
        d = p["delta_days"]
        b = ("1-7d" if d <= 7 else "8-30d" if d <= 30 else "31-90d" if d <= 90 else
             "91-180d" if d <= 180 else "181-365d" if d <= 365 else ">365d")
        buckets[b] += 1
    order = ["1-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">365d"]
    print("\n  delta_days distribution:")
    for b in order:
        n = buckets.get(b, 0)
        bar = "#" * int(40 * n / max(len(pairs), 1))
        print(f"    {b:>9}: {n:5d}  {bar}")

    # ---- write manifest ----
    fields = ["patient", "prior_volume", "curr_volume", "prior_date", "curr_date",
              "delta_days", "patient_age", "patient_sex", "manufacturer_match",
              "model_match", "kernel_match", "curr_has_comparison",
              "prior_findings", "prior_impression", "curr_findings",
              "curr_impression", "curr_clinical"]
    out = os.path.join(OUT_DIR, "ctrate_pairs_enriched.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow(p)
    print(f"\nwrote enriched manifest -> {out}  ({len(pairs):,} pairs)")

    summ = os.path.join(OUT_DIR, "ctrate_enrich_summary.txt")
    with open(summ, "w") as f:
        f.write(f"pairs_dated: {len(pairs)}\n")
        f.write(f"dropped_sameday: {dropped_sameday}\n")
        f.write(f"dropped_missing_date: {dropped_nodate}\n")
        if pairs:
            f.write(f"comparison_frac_clean: {cmp_frac:.4f}\n")
        med = sorted(p["delta_days"] for p in pairs)
        if med:
            f.write(f"delta_days_median: {med[len(med)//2]}\n")
            f.write(f"delta_days_min: {med[0]}\n")
            f.write(f"delta_days_max: {med[-1]}\n")
    print(f"wrote summary -> {summ}")


if __name__ == "__main__":
    main()
