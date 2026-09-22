"""Synthetic, standard-library-only tests; no reports, weights, or network needed."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import preflight_labels as preflight


def synthetic_records(lastblock=False):
    records = []
    for sp in ('train', 'tune', 'test'):
        patient = next(f'SYNTHETIC_{i}' for i in range(1000)
                       if preflight.patient_partition(f'SYNTHETIC_{i}') == ('train' if sp == 'test' else sp))
        if sp == 'test':
            patient += '_TEST'
        prefix = 'valid' if sp == 'test' else 'train'
        finding = dict(finding='Pleural effusion', direction='worsened', tier='explicit',
                       evidence='SYNTHETIC EXAMPLE: effusion increased.')
        if lastblock:
            finding.pop('tier')
            finding.update(progression_eligible=True, original_label_source='report_explicit',
                           label_source='report_explicit', report_grounded=True,
                           training_temporal_sentence='SYNTHETIC EXAMPLE: effusion worsened.')
        records.append(dict(patient=patient, prior_volume=f'{prefix}_{patient}_a_1.nii.gz',
                            curr_volume=f'{prefix}_{patient}_b_1.nii.gz', delta_days=30,
                            parse_ok=True, findings=[finding]))
    return records


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'labels.jsonl'

    def write(self, records):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        return self.path

    def check(self, records, notebook='01', **kwargs):
        return preflight.validate_labels(self.write(records), notebook, **kwargs)

    def test_frozen_valid_both_notebooks(self):
        for notebook in ('01', '02'):
            result = self.check(synthetic_records(), notebook)
            self.assertEqual(result['status'], 'PASS_CHECKED_SCOPE_ONLY')
            self.assertEqual(dict(result['split_findings']), dict(train=1, tune=1, test=1))

    def test_source_aware_is_not_dropin(self):
        result = self.check(synthetic_records(True))
        self.assertEqual(result['errors']['missing_or_invalid_tier'], 3)

    def test_lastblock_schema_vs_exact(self):
        rows = synthetic_records(True)
        self.assertEqual(self.check(rows, '03', schema_only=True)['status'], 'PASS_CHECKED_SCOPE_ONLY')
        result = self.check(rows, '03')
        self.assertIn('historical_lastblock_hash_mismatch', result['errors'])
        self.assertIn('historical_pair_counts_mismatch', result['errors'])

    def test_lastblock_requires_text(self):
        rows = synthetic_records(True)
        rows[0]['findings'][0]['training_temporal_sentence'] = ''
        self.assertIn('missing_training_text_for_eligible_finding',
                      self.check(rows, '03', schema_only=True)['errors'])

    def test_source_fallback_excluded(self):
        rows = synthetic_records(True)
        fallback = copy.deepcopy(rows[0]['findings'][0])
        fallback.update(finding='Lung nodule', label_source='structured_presence',
                        original_label_source='structured_presence', report_grounded=False)
        rows[0]['findings'].append(fallback)
        result = self.check(rows, '03', schema_only=True)
        self.assertEqual(result['exclusions']['not_report_explicit'], 1)
        self.assertEqual(result['split_findings']['train'], 1)

    def test_duplicate_pair(self):
        rows = synthetic_records()
        self.assertIn('duplicate_label_pair', self.check(rows + [rows[0]])['errors'])

    def test_unknown_finding_direction_and_boolean(self):
        for field, value, error in [('finding', 'invalid', 'unknown_finding_name'),
                                     ('direction', 'worse', 'invalid_direction')]:
            rows = synthetic_records()
            rows[0]['findings'][0][field] = value
            self.assertIn(error, self.check(rows)['errors'])
        rows = synthetic_records()
        rows[0]['parse_ok'] = 'False'
        self.assertIn('parse_ok_missing_or_not_boolean', self.check(rows)['errors'])

    def test_leakage(self):
        rows = synthetic_records()
        rows[2]['patient'] = rows[0]['patient']
        self.assertEqual(self.check(rows)['errors']['patients_in_multiple_splits'], 1)

    def test_feature_coverage_never_unpickles(self):
        rows = synthetic_records()
        feature_dir = self.root / 'features'
        feature_dir.mkdir()
        for row in rows:
            for key in ('prior_volume', 'curr_volume'):
                (feature_dir / row[key].replace('.nii.gz', '.pt')).write_bytes(b'NOT A PICKLE')
        result = self.check(rows, feature_dir=feature_dir)
        self.assertEqual(result['status'], 'PASS_CHECKED_SCOPE_ONLY')
        self.assertEqual(result['feature_coverage']['present'], 6)
        next(feature_dir.iterdir()).write_bytes(b'')
        self.assertIn('missing_or_empty_feature_files', self.check(rows, feature_dir=feature_dir)['errors'])

    def test_bad_json_private_errors_and_exit_code(self):
        self.path.write_text('SECRET_REPORT_TEXT not json\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = preflight.main(['labels', '--notebook', '01', '--labels', str(self.path)])
        self.assertEqual(code, 1)
        self.assertNotIn('SECRET_REPORT_TEXT', out.getvalue())
        self.assertNotIn(str(self.path), out.getvalue())

    def test_merlin_splits(self):
        rows = synthetic_records()
        split_path = self.root / 'splits.csv'
        with split_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[*preflight.PAIR_FIELDS, 'split'])
            writer.writeheader()
            for row, sp in zip(rows, ('train', 'val', 'test')):
                writer.writerow({**{k: row[k] for k in preflight.PAIR_FIELDS}, 'split': sp})
        self.assertEqual(self.check(rows, '06', split_csv=split_path)['status'], 'PASS_CHECKED_SCOPE_ONLY')
        self.assertIn('merlin_requires_split_csv', self.check(rows, '06')['errors'])

    def test_manifest_and_date_validation(self):
        row = dict(patient='SYNTHETIC', prior_volume='train_fake_a_1.nii.gz',
                   curr_volume='train_fake_b_1.nii.gz', delta_days='30',
                   prior_date='2020-01-01', curr_date='2020-01-31',
                   prior_findings='SYNTHETIC report.', prior_impression='',
                   curr_findings='SYNTHETIC report.', curr_impression='',
                   prior_labels='{}', curr_labels='{}', presence_changes='{}')
        path = self.root / 'manifest.csv'
        for delta, status in [('30', 'PASS_CHECKED_SCOPE_ONLY'), ('-1', 'FAIL')]:
            row['delta_days'] = delta
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(preflight.validate_manifest(path)['status'], status)

    def test_optional_hash(self):
        self.assertIn('requested_hash_mismatch', self.check(synthetic_records(), expected_sha256='0'*64)['errors'])

    def test_malformed_record_shapes(self):
        for rows in ([None], [[]], [{'patient': 'private'}]):
            self.assertEqual(self.check(rows)['status'], 'FAIL')
        rows = synthetic_records()
        rows[0]['findings'] = {'not': 'a list'}
        self.assertIn('findings_not_list', self.check(rows)['errors'])

    def test_empty_manifest_and_missing_columns(self):
        path = self.root / 'empty.csv'
        path.write_text('patient\n')
        result = preflight.validate_manifest(path)
        self.assertIn('missing_manifest_columns', result['errors'])
        self.assertIn('empty_manifest', result['errors'])

    def test_lastblock_rejects_final_feature_cache(self):
        result = self.check(synthetic_records(True), '03', schema_only=True, feature_dir=self.root)
        self.assertIn('lastblock_needs_prefix_cache_not_final_vectors', result['errors'])


if __name__ == '__main__':
    unittest.main()