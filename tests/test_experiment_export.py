import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORT = runpy.run_path(str(ROOT / 'scripts/export_experiment.py'))['export']


class ExperimentExportTests(unittest.TestCase):
    def test_committed_archive_hashes_and_no_weights(self):
        root = ROOT / 'reports/train-doocs-001'
        manifest = json.loads((root / 'archive_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['prediction_count'], 20)
        self.assertFalse(manifest['weights_included'])
        for name, item in manifest['files'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), item['sha256'])
            self.assertNotIn(Path(name).suffix, ('.pt', '.bin', '.safetensors'))
        status = json.loads((root / 'run_status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['step'], 1146)

    def test_appended_evaluation_matches_archived_predictions(self):
        root = ROOT / 'reports/train-doocs-001'
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        report = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
        manifest = json.loads((root / 'evaluation_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['evaluation_sha256'], digest(root / 'evaluation.json'))
        self.assertEqual(manifest['initial_archive_manifest_sha256'], digest(root / 'archive_manifest.json'))
        self.assertEqual(report['predictions_sha256'], digest(root / 'predictions.jsonl'))
        self.assertEqual(report['suite_sha256'], digest(ROOT / 'datasets/benchmarks/doocs-leetcode/executable_v1.jsonl'))
        self.assertEqual(report['passed'], sum(r['status'] == 'passed' for r in report['results']))
        self.assertEqual(report['total'], len(report['results']))

    def test_in_progress_export_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('run_manifest.json', 'run_status.json', 'metrics.json', 'training_log.jsonl'):
                (root / name).write_text(json.dumps({'status': 'training'}))
            with self.assertRaisesRegex(ValueError, '完成'):
                EXPORT(root, root / 'archive')
            self.assertFalse((root / 'archive').exists())

    def test_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, '已存在'):
                EXPORT('not-needed', tmp)


if __name__ == '__main__':
    unittest.main()
