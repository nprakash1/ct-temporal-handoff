# Onboarding slides — CT temporal progression

[Download the seven-page PDF](onboarding_visuals.pdf) · [Preview all slides](contact_sheet.png)

**Reviewed against the handoff on 2026-09-22.** Metrics are the historical saved
results, not new reruns. The unresolved VQ-gradient issue remains unresolved.
These visuals explain the current handoff accurately; they do not certify a
working fine-tuning method or image-verified labels.

Seven 16:9 slides are supplied as 3200×1800 PNGs and editable-vector SVGs. There
is no editable PowerPoint file. Slide IDs deliberately retain their old order;
**slide numbers are not notebook numbers**.

## Suggested presentation order

**01 overview → 06 preprocessing → 07 frozen architecture → 03 losses →
02 historical results → 04 last-block experiment → 05 next steps.**

## Slides, notebook mapping and speaker notes

### 01 — Project overview

[PNG](01_project_overview.png) · [SVG](01_project_overview.svg)

“We adapt a CheXTemporal-inspired progression task to paired CT examinations.
For a specified finding, we predict worsened, stable, or improved. New/worse are
merged, as are improved/resolved. Spatial grounding and lesion matching are not
implemented by this classifier. The scan stacks in this figure are schematic,
not patient images.”

### 06 — CT-RATE preprocessing

[PNG](06_ctrate_preprocessing.png) · [SVG](06_ctrate_preprocessing.svg)

Maps to [feature-extraction notebook 04](../../notebooks/support/04_extract_frozen_ctclip_features_all_splits.ipynb)
and [last-block notebook 03](../../notebooks/results/03_ctclip_lastblock_ce_bce_cosine_VQ_ISSUE_executed.ipynb).

“There are two tracks: building dated scan pairs and standardizing each volume.
The local source audit counted 50,188 volume/report rows and 21,304 patients—not
the final temporal cohort. Multiple reconstructions are not independent exams.
Each scan is clipped to −1000 through +200 HU, scaled to −1 through +1, resampled
to 0.75 × 0.75 × 1.5 mm, then cropped/padded to 480 × 480 × 240. Padding −1 is
air-like after normalization, not −1 HU. Frozen adapter training uses final
512-D scan vectors; partial unfreezing requires larger intermediate prefixes.
There is no inter-scan registration or explicit orientation canonicalization.
Consecutive-date pairing does not prove the report compares to that exact prior.”

Counts originate in the original project's metadata audit, not a dataset recount
in this handoff. Preprocessing assumes loaded NIfTI intensities are HU. The two
implementations should not be treated as verified feature-equivalent merely
because their stated spacing/shape/window settings match.

### 07 — Frozen CT-CLIP architecture

[PNG](07_frozen_ctclip_architecture.png) · [SVG](07_frozen_ctclip_architecture.svg)

Maps to [frozen training notebook 01](../../notebooks/results/01_frozen_ctclip_ce_bce_crossmodal_supcon_sweep_executed.ipynb)
and [wording notebook 02](../../notebooks/results/02_frozen_ctclip_heldout_wording_loss_sweep_executed.ipynb).

“Both CT-CLIP towers remain frozen. We encode each scan once, cache its 512-D
vector, and learn a finding-conditioned Difference Transformer. Prior/current
role embeddings distinguish scan order. A change token, conditioned on the
finding, attends to both scan tokens. Its projected embedding is compared with
three frozen text prototypes to predict direction. This is supervised temporal
learning on frozen representations, not zero-shot inference.”

“The lower branches are training-only: CE uses prototype logits; auxiliary BCE
uses a scalar head on the pre-projection change-token state; masked cross-modal
SupCon aligns the projected embedding with text candidates. Same finding and
direction are positives; same finding and other direction are negatives; other
findings are ignored. BCE predicts changed versus stable, not physical size.
Losses are enabled/disabled across ablations. No patient report is required at
inference, only the scan vectors, finding ID and cached prototypes.”

### 03 — Loss comparison

[PNG](03_loss_comparison.png) · [SVG](03_loss_comparison.svg)

“CE selects the correct direction prototype. Instance InfoNCE retrieves a paired
text, potentially treating other correct descriptions as negatives. Masked
cross-modal SupCon instead groups texts by finding and direction. The last-block
experiment uses paired cosine alignment—not SupCon. Cosine alignment pulls
toward a target without a contrastive denominator.”

The older figure calling embedding-only SupCon “Old InfoNCE” is not used here.

### 02 — Historical results

[PNG](02_historical_results.png) · [SVG](02_historical_results.svg)

| Panel | Handoff evidence | Test macro-F1 |
|---|---|---|
| MERLIN | [Notebook 06](../../notebooks/historical/06_merlin_temporal_adapter_and_zeroshot_executed.ipynb) | Zero-shot 0.352; adapter 0.398; 252 test findings |
| Frozen CT-CLIP sweep | [Notebook 01](../../notebooks/results/01_frozen_ctclip_ce_bce_crossmodal_supcon_sweep_executed.ipynb) | CE 0.562; CE+BCE 0.575; SupCon-only 0.473; all three 0.473; 1,288 test findings |
| Held-out wording | [Notebook 02](../../notebooks/results/02_frozen_ctclip_heldout_wording_loss_sweep_executed.ipynb) | CE 0.538; CE+SupCon 0.518; CE+BCE 0.515; all three 0.483; 1,288 test findings |

“These panels are not one pooled leaderboard. Cohorts, prototypes, seeds and
sampling differ. CE-based models performed well in these runs, but the results
do not isolate the effect of SupCon or prove it harmful. Held-out wording scores
average three evaluation resamplings, not three independent training seeds.”

The frozen sweep uses report-derived prototypes and 5,055 training findings. The
wording experiment uses training templates and separate evaluation wording.

### 04 — Intended last-block adaptation and actual status

[PNG](04_current_architecture.png) · [SVG](04_current_architecture.svg)

Maps to [last-block notebook 03](../../notebooks/results/03_ctclip_lastblock_ce_bce_cosine_VQ_ISSUE_executed.ipynb).

“Most of the image encoder remains frozen and its prefix output is cached. The
intention is to train the final depth block and output norm, then use the
Difference Transformer to compare examinations. CTViT's temporal/depth blocks
operate within one scan, not across examination dates. The loss is CE + 0.5 BCE
+ 0.5 paired cosine alignment.”

“The red VQ block is a known issue: in the pinned library, eval mode blocks
upstream visual gradients. The downstream adapter can still learn. The saved
run reached tune macro-F1 0.578 and test 0.525, best epoch 3, on 795 test findings.
This is not evidence of successful image-encoder fine-tuning. An isolated check
confirmed the library behavior, but saved visual-weight deltas were not inspected.
Fix gradient flow without updating the frozen codebook, verify weight updates,
and rerun a matched comparison. Do not compare 0.575 versus 0.525 as a controlled
frozen-versus-unfrozen ablation.”

### 05 — Next experiments

[PNG](05_next_experiments.png) · [SVG](05_next_experiments.svg)

“First verify label validity and exact prior-study correspondence, then establish
an image-reviewed evaluation subset. Compare finding-only/current-only controls
with paired models to demonstrate that both scans matter. Match cohorts,
prototypes, seeds and sampling when comparing losses. Verify encoder updates
before claiming fine-tuning. EXACT, localization and acquisition robustness are
future comparisons, not demonstrated improvements.”

## Labeling clarification for the recipient

The seven slides are a model/results overview, not a complete label-schema guide.
Read [Artifacts and schemas](../../ARTIFACTS_AND_SCHEMAS.md) for the full mapping:

- Frozen runs 01/02 and MERLIN 06 use **historical paired-report comparison labels**
  (legacy artifact suffix `v3`), retaining entries marked explicit.
- Last-block run 03 uses **filtered report-comparison labels plus standardized
  alignment text** (legacy `v6_synthetic_direction`). Templates do not validate labels.
- **Current-report-only labeling**, notebook 05 (legacy `v7`), is optional future
  work and was not used for these historical training results. It extracts
  comparisons already written in a current report, not progression inferred by
  independently comparing images. Structured-presence fallback is marked separately.
- No report-derived label is image-verified gold. Exact report-to-paired-prior
  correspondence remains unverified by the labeling code.

## What was updated for this repository?

- Retained the explicit BCE and SupCon branches in the frozen diagram.
- Reworded “current/latest” as “last-block experiment” where ambiguous.
- Corrected preprocessing to distinguish final vector caches from prefix caches.
- Made the report/prior correspondence caveat explicit on the preprocessing slide.
- Linked every experiment to the descriptively named handoff notebooks.
- Added these saved speaker notes and portable rebuild instructions.

## Rebuild and verify

The generator uses the project's existing **matplotlib** and **Pillow** libraries.
From the repository root in an environment with those dependencies:

```bash
python3 -B tools/make_onboarding_visuals.py
```

By default, it checks the displayed metrics against saved outputs in this
repository's published notebooks before rendering. It records those notebook
hashes/cell indices in [results_provenance.json](results_provenance.json).
The hashes refer to publication snapshots, not the original pre-filtering files.
The publication manifest separately records original/published hashes.

An optional `--source-dir` can point to the original executed notebook folder;
that mode expects the original filenames and records their hashes instead.
`--output-dir` changes the destination. No training or dataset access occurs.

Layout checks detect slide/box text overflow; PNG dimensions are asserted.
Assets contain schematic graphics and aggregate metrics, not patient images,
reports, scan identifiers or credentials. No training code was changed.