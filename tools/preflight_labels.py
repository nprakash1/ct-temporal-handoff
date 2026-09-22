#!/usr/bin/env python3
"""Read-only label/manifest contract checks using only the Python standard library.

Prints aggregate diagnostics, never report text or patient/scan identifiers.
No model execution, pickle loading, rewriting, network access, or clinical validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

FINDINGS = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
    'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
    'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening',
]
DIRECTIONS = {'worsened', 'stable', 'improved'}
PAIR_FIELDS = ('patient', 'prior_volume', 'curr_volume')
LASTBLOCK_SHA = '04ca9952eeb365a3e9bebd7e96f58a82af22f1bf78d06e25eaabd90261457e66'
LASTBLOCK_PAIRS = {'train': 3172, 'tune': 555, 'test': 233}
LASTBLOCK_FINDINGS = {'train': 10603, 'tune': 1882, 'test': 795}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def report(kind):
    return dict(check=kind, errors=Counter(), warnings=Counter(), counts=Counter(),
                exclusions=Counter(), label_sources=Counter(), split_pairs=Counter(),
                split_findings=Counter(), split_directions=defaultdict(Counter),
                scope='Structural checks only; no clinical, report-reference, tensor-content, '
                      'checkpoint, or feature-provenance verification.')


def text(value):
    return isinstance(value, str) and bool(value.strip())


def pair_key(row, result):
    if not isinstance(row, dict) or any(not text(row.get(k)) for k in PAIR_FIELDS):
        result['errors']['missing_or_invalid_pair_fields'] += 1
        return None
    for k in ('prior_volume', 'curr_volume'):
        if '/' in row[k] or '\\' in row[k]:
            result['errors']['scan_id_not_basename'] += 1
            return None
    if row['prior_volume'] == row['curr_volume']:
        result['errors']['identical_prior_current_scan'] += 1
    return tuple(row[k] for k in PAIR_FIELDS)


def patient_partition(patient):
    u = int.from_bytes(hashlib.sha256(f'2026|{patient}'.encode()).digest()[:8], 'big') / 2**64
    return 'tune' if u < .15 else 'train'


def domain_split(row):
    pv, cv = row['prior_volume'], row['curr_volume']
    if pv.startswith('train_') and cv.startswith('train_'):
        return patient_partition(row['patient'])
    if pv.startswith('valid_') and cv.startswith('valid_'):
        return 'test'
    return None


def load_splits(path, result):
    splits = {}
    if path is None:
        result['errors']['merlin_requires_split_csv'] += 1
        return splits
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set((*PAIR_FIELDS, 'split')) <= set(reader.fieldnames or []):
            result['errors']['split_csv_missing_columns'] += 1
            return splits
        for row in reader:
            key = pair_key(row, result)
            if key is None:
                continue
            if key in splits:
                result['errors']['duplicate_split_pair'] += 1
            sp = row['split']
            if sp not in {'train', 'val', 'test'}:
                result['errors']['invalid_split_name'] += 1
                continue
            splits[key] = sp
    return splits


def validate_labels(path, notebook, *, split_csv=None, feature_dir=None,
                    schema_only=False, expected_sha256=None):
    result = report(f'labels_for_notebook_{notebook}')
    result['sha256'] = digest(path)
    lastblock = notebook == '03'
    result['schema_only'] = schema_only
    if expected_sha256 and result['sha256'] != expected_sha256:
        result['errors']['requested_hash_mismatch'] += 1
    if lastblock and not schema_only and result['sha256'] != LASTBLOCK_SHA:
        result['errors']['historical_lastblock_hash_mismatch'] += 1
    if schema_only:
        result['warnings']['schema_only_not_historical_reproduction'] += 1
    if lastblock and feature_dir is not None:
        result['errors']['lastblock_needs_prefix_cache_not_final_vectors'] += 1
        feature_dir = None
    splits = load_splits(split_csv, result) if notebook == '06' else {}
    patients = defaultdict(set)
    volumes = set()
    seen = set()
    feature_present = set()
    feature_missing = set()
    available_pairs = Counter()
    available_findings = Counter()
    if feature_dir is not None and not Path(feature_dir).is_dir():
        result['errors']['feature_directory_missing'] += 1
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            result['counts']['records'] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result['errors']['invalid_json_line'] += 1
                continue
            key = pair_key(row, result)
            if key is None:
                continue
            if key in seen:
                result['errors']['duplicate_label_pair'] += 1
            seen.add(key)
            if type(row.get('parse_ok')) is not bool:
                result['errors']['parse_ok_missing_or_not_boolean'] += 1
            fs = row.get('findings')
            if not isinstance(fs, list):
                result['errors']['findings_not_list'] += 1
                continue
            try:
                gap = int(row.get('delta_days'))
                if gap <= 0:
                    result['warnings']['nonpositive_interval'] += 1
            except (ValueError, TypeError):
                result['errors']['invalid_or_missing_delta_days'] += 1
            eligible = []
            finding_seen = set()
            for finding in fs:
                result['counts']['findings'] += 1
                if not isinstance(finding, dict):
                    result['errors']['finding_not_object'] += 1
                    continue
                name, direction = finding.get('finding'), finding.get('direction')
                if not isinstance(name, str) or name not in FINDINGS:
                    result['errors']['unknown_finding_name'] += 1
                    continue
                if name in finding_seen:
                    result['errors']['duplicate_finding_in_pair'] += 1
                finding_seen.add(name)
                if not isinstance(direction, str) or direction not in DIRECTIONS | {'unknown'}:
                    result['errors']['invalid_direction'] += 1
                    continue
                source = finding.get('label_source') if lastblock else finding.get('tier')
                result['label_sources'][source if isinstance(source, str) else '(missing)'] += 1
                if lastblock:
                    required = ('progression_eligible', 'original_label_source', 'label_source',
                                'report_grounded', 'training_temporal_sentence')
                    if any(k not in finding for k in required):
                        result['errors']['missing_lastblock_finding_fields'] += 1
                        continue
                    if any(type(finding[k]) is not bool for k in ('progression_eligible', 'report_grounded')):
                        result['errors']['invalid_lastblock_boolean'] += 1
                        continue
                    if not finding['progression_eligible']:
                        result['exclusions']['not_progression_eligible'] += 1
                        continue
                    if finding['original_label_source'] != 'report_explicit' or finding['label_source'] != 'report_explicit':
                        result['exclusions']['not_report_explicit'] += 1
                        continue
                    if not finding['report_grounded']:
                        result['exclusions']['not_report_grounded'] += 1
                        continue
                else:
                    if finding.get('tier') not in ('explicit', 'inferred'):
                        result['errors']['missing_or_invalid_tier'] += 1
                        continue
                    if finding['tier'] != 'explicit':
                        result['exclusions']['not_explicit'] += 1
                        continue
                if direction not in DIRECTIONS:
                    result['exclusions']['unknown_direction'] += 1
                    continue
                if lastblock and not text(finding['training_temporal_sentence']):
                    result['errors']['missing_training_text_for_eligible_finding'] += 1
                    continue
                if not lastblock and not text(finding.get('evidence')):
                    result['warnings']['eligible_finding_without_evidence_text'] += 1
                eligible.append(direction)
            if notebook != '06' and row.get('parse_ok') is not True:
                result['exclusions']['parse_failed_records'] += 1
                continue
            sp = splits.get(key) if notebook == '06' else domain_split(row)
            if sp is None:
                result['exclusions']['pair_outside_split_or_domain'] += 1
                continue
            if not eligible:
                result['exclusions']['no_eligible_findings'] += 1
                continue
            patients[sp].add(row['patient'])
            result['split_pairs'][sp] += 1
            result['split_findings'][sp] += len(eligible)
            result['split_directions'][sp].update(eligible)
            scans = (row['prior_volume'], row['curr_volume'])
            volumes.update(scans)
            if feature_dir is not None:
                exists = []
                for scan in scans:
                    stem = scan.replace('.nii.gz', '').replace('.nii', '')
                    fp = Path(feature_dir) / (stem + '.pt')
                    ok = fp.is_file() and fp.stat().st_size > 0
                    (feature_present if ok else feature_missing).add(scan)
                    exists.append(ok)
                if all(exists):
                    available_pairs[sp] += 1
                    available_findings[sp] += len(eligible)
    if notebook == '06':
        result['counts']['split_pairs_without_labels'] = len(set(splits) - seen)
        if set(splits) - seen:
            result['warnings']['split_pairs_without_labels'] += len(set(splits) - seen)
    names = ('train', 'val', 'test') if notebook == '06' else ('train', 'tune', 'test')
    for i, sp in enumerate(names):
        if not result['split_findings'][sp]:
            result['errors']['empty_eligible_split'] += 1
        for other in names[i+1:]:
            result['errors']['patients_in_multiple_splits'] += len(patients[sp] & patients[other])
    result['counts']['unique_eligible_scans'] = len(volumes)
    if lastblock and not schema_only:
        if dict(result['split_pairs']) != LASTBLOCK_PAIRS:
            result['errors']['historical_pair_counts_mismatch'] += 1
        if dict(result['split_findings']) != LASTBLOCK_FINDINGS:
            result['errors']['historical_finding_counts_mismatch'] += 1
        if len(volumes) != 6590:
            result['errors']['historical_scan_count_mismatch'] += 1
    if feature_dir is None:
        result['warnings']['feature_coverage_not_checked'] += 1
    else:
        result['feature_coverage'] = dict(present=len(feature_present), missing_or_empty=len(feature_missing),
                                        usable_pairs=dict(available_pairs), usable_findings=dict(available_findings))
        if feature_missing:
            result['errors']['missing_or_empty_feature_files'] += len(feature_missing)
        result['warnings']['feature_contents_and_provenance_not_checked'] += 1
    result['errors'] = Counter({k: v for k, v in result['errors'].items() if v})
    result['status'] = 'FAIL' if result['errors'] else 'PASS_CHECKED_SCOPE_ONLY'
    return result


def validate_manifest(path):
    result = report('labeler_input_manifest')
    result['sha256'] = digest(path)
    required = set(PAIR_FIELDS) | {'delta_days', 'prior_findings', 'prior_impression',
        'curr_findings', 'curr_impression', 'prior_labels', 'curr_labels', 'presence_changes'}
    seen = set()
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            result['errors']['missing_manifest_columns'] += len(required - set(reader.fieldnames or []))
        for row in reader:
            result['counts']['rows'] += 1
            key = pair_key(row, result)
            if key:
                if key in seen:
                    result['errors']['duplicate_manifest_pair'] += 1
                seen.add(key)
            try:
                gap = int(row.get('delta_days', ''))
                if gap <= 0:
                    result['errors']['nonpositive_interval'] += 1
            except (ValueError, TypeError):
                gap = None
                result['errors']['invalid_interval'] += 1
            if not (text(row.get('curr_findings')) or text(row.get('curr_impression'))):
                result['errors']['empty_current_report'] += 1
            if not (text(row.get('prior_findings')) or text(row.get('prior_impression'))):
                result['warnings']['empty_prior_report'] += 1
            for field in ('prior_labels', 'curr_labels', 'presence_changes'):
                raw = row.get(field, '')
                if not raw:
                    result['warnings']['missing_structured_signals'] += 1
                    continue
                try:
                    data = json.loads(raw)
                    allowed = {'new', 'resolved', 'present_both', 'absent_both'} if field == 'presence_changes' else {0, 1}
                    if not isinstance(data, dict) or any(k not in FINDINGS or not isinstance(v, (str, int, float)) or v not in allowed for k, v in data.items()):
                        result['errors']['invalid_structured_mapping'] += 1
                except json.JSONDecodeError:
                    result['errors']['invalid_structured_json'] += 1
            dates = (row.get('prior_date'), row.get('curr_date'))
            if all(dates):
                try:
                    dt = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
                    if (dt[1] - dt[0]).days != gap or dt[1] <= dt[0]:
                        result['errors']['date_interval_mismatch'] += 1
                except (ValueError, TypeError):
                    result['errors']['invalid_iso_date'] += 1
            else:
                result['warnings']['dates_not_available_for_check'] += 1
    if not result['counts']['rows']:
        result['errors']['empty_manifest'] += 1
    result['status'] = 'FAIL' if result['errors'] else 'PASS_CHECKED_SCOPE_ONLY'
    return result


def main(argv=None):
    csv.field_size_limit(10**9)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    labels = sub.add_parser('labels', help='Check training-label compatibility, not generate labels')
    labels.add_argument('--notebook', choices=['01', '02', '03', '06'], required=True)
    labels.add_argument('--labels', type=Path, required=True)
    labels.add_argument('--split-csv', type=Path, help='Required for historical MERLIN (06)')
    labels.add_argument('--feature-dir', type=Path, help='Optional flat per-scan .pt directory; never unpickled')
    labels.add_argument('--schema-only', action='store_true', help='Skip historical 03 hash/count contract; NOT reproduction approval')
    labels.add_argument('--expected-sha256', help='Optional exact hash from artifact manifest')
    manifest = sub.add_parser('manifest', help='Check enriched CSV input to labelers 05/07/08')
    manifest.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'manifest':
            result = validate_manifest(args.manifest)
        else:
            result = validate_labels(args.labels, args.notebook, split_csv=args.split_csv,
                                     feature_dir=args.feature_dir, schema_only=args.schema_only,
                                     expected_sha256=args.expected_sha256)
    except (OSError, UnicodeError, csv.Error):
        # Do not print potentially sensitive paths or raw row content in exceptions.
        result = dict(status='FAIL', errors={'input_unreadable_or_invalid_encoding': 1})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())