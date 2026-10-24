import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'scripts'), *sys.path]):
    HOLDOUT = runpy.run_path(str(ROOT / 'scripts/evaluate_holdout.py'))


class HoldoutTests(unittest.TestCase):
    def test_changed_test_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'test.jsonl'
            source.write_bytes((ROOT / 'datasets/leetcode/archive/codex_v1/test.jsonl').read_bytes())
            (root / 'run_manifest.json').write_text(json.dumps({
                'data_sha256': {'test': hashlib.sha256(source.read_bytes()).hexdigest()}}))
            _, rows = HOLDOUT['load_test'](root, source)
            self.assertEqual(len(rows), 5)
            source.write_bytes(source.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, '哈希'):
                HOLDOUT['load_test'](root, source)

    def test_generation_rejects_overwrite_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(HOLDOUT['generate'].__globals__, {'load_test': lambda *a: ({}, [])}):
                with self.assertRaises(FileExistsError):
                    HOLDOUT['generate']('unused', 'unused', tmp, 512)


if __name__ == '__main__':
    unittest.main()
