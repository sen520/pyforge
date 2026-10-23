import ast
import json
from pathlib import Path
import runpy
import unittest

from pyforge.training_config import load_splits


ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / 'scripts/build_leetcode_training.py'))


class LeetcodeTrainingTests(unittest.TestCase):
    def test_repaired_interfaces_match_actual_answer_and_preserve_hashes(self):
        path = ROOT / 'datasets/metadata/interface_repair_v1/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(len(manifest['changes']), 57)
        changes = {r['id']: r for r in manifest['changes']}
        from hashlib import sha256
        for split in ('train', 'validation'):
            for row in map(json.loads, (ROOT / 'datasets' / (split + '.jsonl')).read_text(encoding='utf8').splitlines()):
                if row['id'] not in changes:
                    continue
                change = changes[row['id']]
                self.assertEqual(sha256(row['messages'][-1]['content'].encode()).hexdigest(), change['answer_sha256_unchanged'])
                self.assertEqual(row['messages'][1]['content'], change['new_user_message'])
                self.assertLessEqual(row['token_length'], 2048)
                interface = ast.parse(change['new_interface']).body[0]
                code = BUILDER['normalize'](row['messages'][-1]['content'].split('```python\n')[1].rsplit('```', 1)[0])
                actual = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == interface.name)
                signatures = {n.name: ast.dump(n.args) for n in actual.body if isinstance(n, ast.FunctionDef)}
                for method in interface.body:
                    if isinstance(method, ast.FunctionDef):
                        self.assertEqual(ast.dump(method.args), signatures[method.name])

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
        current = json.loads((ROOT / 'datasets/metadata/current_manifest.json').read_text(encoding='utf-8'))
        splits, hashes = load_splits(ROOT / 'datasets')
        reserved = set(manifest['reserved_benchmark_ids'])
        for name, rows in splits.items():
            self.assertEqual(len(rows), manifest['files'][name]['count'])
            self.assertEqual(hashes[name], current['files'][name]['sha256'])
            self.assertFalse(reserved & {r['id'] for r in rows})
            for row in rows:
                if row['source'] == 'doocs_reference_sft_v1':
                    self.assertFalse(row['verification']['execution'])
                    self.assertNotEqual(row['author'], 'Codex')
                    ast.parse(row['messages'][-1]['content'].split('```python\n')[1].rsplit('```', 1)[0])


if __name__ == '__main__':
    unittest.main()
