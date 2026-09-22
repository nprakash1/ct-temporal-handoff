# How to use this handoff

## Read first

1. Read the README's notebook index and labeling-method explanations.
2. Start with result notebook 01; use 02 for wording robustness and 03 for the
   unresolved last-block experiment. Do not run every notebook in numerical order.
3. Run the CPU-only checks from the repository root:

```bash
python3 -B tools/verify_handoff.py
python3 -B tools/test_preflight_labels.py
python3 -B tools/preflight_labels.py --help
python3 -B tools/preflight_labels.py manifest --manifest examples/synthetic_labeler_input.csv
```

These tools use the Python standard library. They do not download data or models.

## Before training in Colab

The notebooks are **historical source snapshots** with aggregate result outputs,
not reconfigured runnable notebooks. Preserve them; make a working copy in Colab.

- Obtain the exact compatible labels/features through approved storage. Locations
  in the artifact manifest are the original owner's local paths, not shared URLs.
- For 01/02/04, the source clones `https://github.com/nprakash1/3dCT.git` and imports
  its `scripts/ctclip_utils.py`. If you do not have access, configure your working
  copy to clone this repository and import its `reference_code/ctclip_utils.py`
  instead. Record that helper revision change; historical helper equivalence is
  not established merely by copying the current helper.
- Replace hard-coded label paths with your actual approved artifact locations.
  Changing the clone URL alone does not supply the excluded JSONL files.
- Set your Drive root and separate run/cache directories to avoid overwriting
  historical caches/checkpoints. Do not reuse caches without provenance checks.
- Use notebook-specific GPU/dependency setup. This package does not claim to
  provide one fully tested environment for CT-CLIP, MedGemma and MERLIN together.
- Validate labels with `tools/preflight_labels.py` before training. See the schema
  guide for the correct notebook profile. No arbitrary interchange of label sets.
- The copied `_validate_v7_notebook.py` is a historical reference targeting the
  old notebook path, not a portable entry point. The tests above are portable.
- Notebook 03's VQ gradient problem is unresolved. Repair/verify it in a new
  experiment; its recorded metrics do not demonstrate visual fine-tuning.

## Publication policy

This repository contains no raw scans, label JSONLs, feature tensors or weights.
Notebook code cells were preserved. All outputs except selected aggregate result
cells were removed; only plain-text output representations were retained. Session
metadata and attachments were removed. See `publication_manifest.json` for the
transformation and hashes. Original fully executed evidence remains with the owner.

Limited credential and identifier scans are not a clinical-data privacy audit.
Keep the repository private and review again before any public release. Check
dataset/model access terms and institutional permissions before sharing artifacts.
No license for third-party datasets/models is granted by this repository.