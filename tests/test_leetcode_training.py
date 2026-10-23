import ast
import json
from pathlib import Path
import runpy
import unittest

from pyforge.training_config import load_splits


ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / 'scripts/build_leetcode_training.py'))


class LeetcodeTrainingTests(unittest.TestCase):
    def test_imports_added_without_execution(self):
        code = 'class Solution:\n def f(self, a: List[int]):\n  return Counter(a)\n'
        result = BUILDER['normalize'](code)
        self.assertIn('from typing import List', result)
        self.assertIn('from collections import Counter', result)
        ast.parse(result)

    def test_unsupported_dependencies_rejected(self):
        for code in ('class Solution:\n def f(self): return SortedList()',
                     'from sortedcontainers import SortedList\nclass Solution: pass',
                     'print("must not execute")'):
            with self.assertRaises(ValueError):
                BUILDER['normalize'](code)

    def test_expanded_splits_and_reserved_benchmark(self):
        manifest_path = ROOT / 'datasets/metadata/doocs_sft_v1/manifest.json'
        if not manifest_path.exists():
            self.skipTest('扩充数据尚未构建')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        splits, hashes = load_splits(ROOT / 'datasets')
        reserved = set(manifest['reserved_benchmark_ids'])
        for name, rows in splits.items():
            self.assertEqual(len(rows), manifest['files'][name]['count'])
            self.assertEqual(hashes[name], manifest['files'][name]['sha256'])
            self.assertFalse(reserved & {r['id'] for r in rows})
            for row in rows:
                if row['source'] == 'doocs_reference_sft_v1':
                    self.assertFalse(row['verification']['execution'])
                    self.assertNotEqual(row['author'], 'Codex')
                    ast.parse(row['messages'][-1]['content'].split('```python\n')[1].rsplit('```', 1)[0])


if __name__ == '__main__':
    unittest.main()
