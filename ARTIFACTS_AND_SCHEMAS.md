# Label creation, artifacts, and input contracts

This guide connects **report-pair preparation → label creation → experiment inputs**.
All labels here are report-derived silver supervision, not image-reviewed truth.
The preflight tool verifies data contracts, not clinical correctness or which
prior examination a report references. No artifact containing real report text
was copied into this handoff.

## 1. Which creator matches which training notebook?

| Training experiment | Matching creation workflow | Actual input artifact |
|---|---|---|
| 01 frozen loss sweep, 02 held-out wording, 06 MERLIN | **07: paired-report labeling for frozen baselines** | `medgemma_labels_v3.jsonl` |
| 03 intended last-block adaptation | **08: source-aware paired-report labeling**, then **standardized-text conversion** | `medgemma_labels_v6_synthetic_direction.jsonl` |
| No selected historical training run | **05: current-report-only extraction**, followed by separate structured fallback | `medgemma_labels_v7.jsonl`; needs a new loader/experiment before training |
| 04 feature extraction | No label creator required | CT volumes; writes embeddings, not labels |

```text
Official reports + metadata → date-ordered pair manifest
Official predicted presence labels → enriched pair manifest
                                       |
              +------------------------+---------------------+
              |                        |                     |
       07 paired reports      08 paired reports +      05 current report
       text precedence        structured context       only to the LLM
              |                        |                     |
       older label schema     provenance-aware labels   future dataset
              |                        |                 (not plug-and-play)
         01 / 02 / 06         standardized-text script
                                       |
                               03's filtered cohort
```

### Notebook 07 — historical paired-report baseline labeler

[Open 07](notebooks/label_creation/07_create_paired_report_labels_for_frozen_baselines.ipynb)

Source: `/Users/nealprakash/3dCT/notebooks/medgemma_label_colab.ipynb`, copied without
changes. It supplies both reports, asks for up/down/same/not-mentioned for the 18
findings, and combines results with structured presence transitions afterward.
Outputs marked `tier = explicit` supply the baseline training labels.

**Historical limitation:** its worked example allows changes inferred from static
prior/current descriptions despite instructions requiring explicit comparison.
An `explicit` tag therefore does not prove a true temporal sentence or image change.
Do not silently repair this prompt and attribute regenerated labels to the old run.

Configured output: `/content/drive/MyDrive/ct_temporal/medgemma_labels_v3.jsonl`.
No model weights are trained or saved; MedGemma is used for inference.

### Notebook 08 — historical source-aware paired-report labeler

[Open 08](notebooks/label_creation/08_create_source_aware_paired_report_labels.ipynb)

Recovered byte-for-byte from commit
`2e683da873d766216c8eefbc0f903a953383fc7f`, historical repository path
`/Users/nealprakash/3dCT/notebooks/label_per_finding_temporal_colab.ipynb`.
Its prompt sees both reports, interval, finding list, and noisy structured
transitions. It requests complete verbatim temporal sentences and records
report-explicit versus inferred/fallback sources separately. Both-scan presence
without comparison is unknown, never automatically stable.

Configured output: `/content/drive/MyDrive/ct_temporal/medgemma_labels_v6.jsonl`.
The available local source artifact is named
`/Users/nealprakash/3dCT/medgemma_labels_v6 (1).jsonl`.
No MedGemma fine-tuning occurs. This recovered revision documents the historical
method; the exact original generation revision/runtime has not been independently
proven from that artifact. Rerunning an LLM is not guaranteed to reproduce its bytes.

Next run the copied helper
`./reference_code/21_synthesize_v6_direction_text.py`.
It preserves original fields and adds a deterministic label-matched sentence:

```bash
python3 -B ./reference_code/21_synthesize_v6_direction_text.py --input '/Users/nealprakash/3dCT/medgemma_labels_v6 (1).jsonl' --output /tmp/ct_progression_synthetic_direction_recreated.jsonl
```

The helper marks known directions eligible when record parse/schema checks pass;
**notebook 03 adds stricter report-explicit/grounding filters**. The converter
itself can add text to presence-fallback labels. A synthetic sentence is not new
clinical evidence, and eligibility from the converter alone is not 03 eligibility.
Use a new destination; do not overwrite the original evidence artifact.

## 2. Artifact access and hashes

`./artifact_manifest.json` contains byte counts,
SHA-256 hashes, current local source locations, purposes and consumers for:

- Frozen-baseline label JSONL.
- Optional historical dynamic-sentence JSONL used by 01/02 (not their direction labels).
- Source-aware labels before conversion.
- Standardized-direction labels used by 03.
- Enriched report-pair manifest.
- Historical MERLIN split CSV.

Only metadata is included. **No approved remote sharing URL has been configured.**
The owner must arrange access-controlled transfer permitted by the dataset terms.
The local source paths will not exist on a recipient's machine. Do not commit these
report-bearing artifacts or CT volumes to a public repository.

The labeler output locations above are configured destinations, not proof the
artifacts still exist on Drive. The local file hashes identify bytes inspected
here; they do not establish clinical accuracy. Notebook 03 independently asserts
the exact standardized-label hash in its own code. Older experiments do not pin
their label files by hash: local hashes should not be described as proof of every
historical run's input bytes.

Supply the original artifacts for exact historical inputs even when supplying
creation code. LLM/version/dependency changes can produce different labels.

## 3. Input to label-creation notebooks 05, 07 and 08

They consume an enriched CSV, historically named `ctrate_pairs_enriched_v2.csv`.
Its `v2` filename describes a manifest revision, not the labeling method.

| Field | Type / meaning |
|---|---|
| `patient` | Nonempty patient/group identifier for grouping and disjoint splits |
| `prior_volume`, `curr_volume` | Exact scan basenames, normally `.nii.gz`; distinct examinations |
| `delta_days` | Positive integer interval |
| `prior_findings`, `prior_impression` | Prior report sections; may have an empty section |
| `curr_findings`, `curr_impression` | Current report sections; at least one must contain text |
| `prior_labels`, `curr_labels` | JSON objects in CSV cells, mapping canonical finding names to predicted 0/1 presence |
| `presence_changes` | JSON object: finding → `new`, `resolved`, `present_both`; absent-both is generally omitted |
| `prior_date`, `curr_date` | Recommended ISO `YYYY-MM-DD` dates; the preflight checks agreement with `delta_days` when supplied |
| `labels_found` | Audit flag from the join: 1 if both scans had structured labels, otherwise 0 |

Other metadata columns are allowed. Blank structured cells are warned about:
silently treating unavailable predictions as absent-both is not reliable evidence.
All three labelers use the structured fields for postprocessing; **05 does not
expose them or the prior report to the LLM when judging explicit semantics**.

Copies of the preparation scripts are included:

1. `./reference_code/12_ctrate_enrich.py`: fetch official
   report/metadata CSVs, pick reconstructions, order by dates, pair consecutive
   studies, drop same-day/missing-date pairs; writes the initial enriched CSV.
2. `./reference_code/16_join_abnormality_labels.py`:
   joins official predicted presence labels, adds the JSON columns and writes
   the enriched input consumed by the labelers.

Both use paths relative to the working directory and may download gated data.
Run them in a separate approved workspace with `HF_TOKEN` supplied securely;
they are **not** run automatically by the read-only validator. Pin upstream
dataset revisions for a new reproducible study. The historical script's header
describing an older severity-only LLM workflow is not the specification of 07/08.

Synthetic manifest example:
`./examples/synthetic_labeler_input.csv`.
These invented IDs cannot fetch real scans; this is a schema example only.

## 4. Training-label JSONL contracts

Each nonblank line is one JSON object per `(patient, prior_volume, curr_volume)`.
Records must have `findings` as a list and `delta_days` as an integer-compatible
interval. Preserve IDs exactly so labels join to splits and scan features.

### Frozen-baseline contract — notebooks 01/02/06

- Record fields: `patient`, `prior_volume`, `curr_volume`, `delta_days`,
  `parse_ok` (Boolean), `findings` (list).
- Finding fields: `finding` (one of the 18 canonical names), `direction`
  (`worsened`, `stable`, `improved`, or `unknown`), `tier` (`explicit` or `inferred`).
- `evidence` is a report quote used by text-target construction in 01/02; absence
  is warned about. Optional dynamic text is read separately and may affect targets.
- Training retains `tier = explicit` and a known direction. 01/02 additionally
  require `parse_ok`, paired feature coverage, and recognized scan domains.
- 06 uses a separate CSV with `patient, prior_volume, curr_volume, split` where
  `split` is `train`, `val`, or `test`. Its loader does not filter on `parse_ok`;
  the validator still checks that the field has the expected Boolean type.
- 01/02 use `train_*` pairs for train/tune, `valid_*` pairs for test; train/tune is
  patient-hashed with seed 2026 and 15% tune. Cross-domain pairs are excluded.

The validator reports candidate counts **before feature coverage** plus usable
counts if a feature directory is provided. It intentionally checks patient
disjointness on all eligible candidates (conservative, before missing features).
A duplicate pair is an error even if old code would silently overwrite it.

### Last-block contract — notebook 03

In addition to the pair fields and record `parse_ok`, each finding needs:

- `finding`, `direction`.
- `progression_eligible` (Boolean).
- `original_label_source`, `label_source`.
- `report_grounded` (Boolean; denotes text checks, not image verification).
- `training_temporal_sentence` (nonempty for accepted training findings).

Retained findings require eligibility, both source fields `report_explicit`,
grounding true, and a known direction/name. The loader uses the same patient hash
and scan-domain split as above. It also requires this exact historical contract:

- SHA-256: `04ca9952eeb365a3e9bebd7e96f58a82af22f1bf78d06e25eaabd90261457e66`
- Pair counts train/tune/test: **3172 / 555 / 233**.
- Finding counts train/tune/test: **10603 / 1882 / 795**.
- Unique scans: **6590**.

Passing `--schema-only` skips only these fixed hash/count requirements. It does
**not** make new labels runnable in the unchanged notebook, remove that notebook's
assertions, or certify reproduction. Revise loaders/configuration explicitly for
a new experiment; do not fabricate provenance fields merely to pass a filter.

### New current-report-only labels from 05

They contain provenance fields such as `label_source`, `report_grounded`,
`temporal_sentence_source`, not the older `tier` schema. They also lack the
standardized-text derivative fields expected by 03. Therefore they are not a
drop-in input to any completed training experiment. The preflight intentionally
has no “accept anything” profile and does not convert labels.

## 5. Read-only preflight commands

Tool: `./tools/preflight_labels.py`.
Uses only Python's standard library. Exit **0** means the checked scope passed;
exit **1** means errors. JSON output contains aggregate counts and fixed error
codes, not report text, patient identifiers, scan filenames or raw exceptions.

Check historical frozen labels (repeat with `--notebook 02` as appropriate):

```bash
python3 -B ./tools/preflight_labels.py labels --notebook 01 --labels /Users/nealprakash/3dCT/medgemma_labels_v3.jsonl
```

Check exact last-block inputs:

```bash
python3 -B ./tools/preflight_labels.py labels --notebook 03 --labels /Users/nealprakash/3dCT/medgemma_labels_v6_synthetic_direction.jsonl
```

Check historical MERLIN labels and splits:

```bash
python3 -B ./tools/preflight_labels.py labels --notebook 06 --labels /Users/nealprakash/3dCT/medgemma_labels_v3.jsonl --split-csv /Users/nealprakash/3dCT/data/ctrate/subset_pairs.csv
```

Check input to the label creators:

```bash
python3 -B ./tools/preflight_labels.py manifest --manifest /Users/nealprakash/3dCT/data/ctrate/ctrate_pairs_enriched_v2.csv
```

For 01/02/06, optional `--feature-dir` takes an absolute path to a **flat per-scan
`.pt` cache**. It checks filenames/existence/nonzero size only; never unpickles
tensors. Missing files fail preflight even if training could skip them. Directory
coverage does not verify tensor shape, numerical values, model revision or
preprocessing provenance. 03 rejects this option because final vectors cannot
stand in for its larger prefix cache. Optional `--expected-sha256` checks an exact
artifact hash for any supported training profile.

Without features, passing means **label contract only**, not training readiness.
Prototype coverage (especially MERLIN), optional dynamic-text contents, semantic
text/label agreement, checkpoint compatibility, and actual image/reference
correspondence remain outside this tool's scope. No GPU training was performed.

## 6. Tests and observed local checks

```bash
python3 -B ./tools/test_preflight_labels.py
```

Tests use invented records and cover both frozen profiles, schema mismatch,
last-block exact versus schema-only mode, missing text, fallback exclusion,
duplicates, invalid names/directions/booleans, patient leakage, feature coverage,
malformed JSON, redacted diagnostics, MERLIN splits, dates, and expected hashes.
There are 16 standard-library unit tests. To verify the publication notebook
hashes, source-cell syntax and documentation links on the recipient's machine:

```bash
python3 -B ./tools/verify_handoff.py
```

This verifier does not require the original local files or Git history. Original
hashes and historical paths remain in the metadata manifests for provenance.
The recipient-facing label preflight accepts the recipient's own artifact paths.

Local 03 inputs matched the exact hash/cohort contract. The local enriched
manifest passed its structural checks. Local older labels yield **17,828 train /
3,106 tune / 1,288 test finding candidates before feature filtering**, with four
eligible findings lacking evidence text. This does not reproduce the historical
5,055/899 training/tune feature-covered cohort without its exact cache coverage.
The tool reports the difference rather than hiding it.