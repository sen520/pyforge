import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPARE = runpy.run_path(str(ROOT / 'scripts/compare_eval.py'))['compare']


class ComparisonTests(unittest.TestCase):
    def inputs(self):
        root = ROOT / 'reports/train-doocs-001'
        return [ROOT / 'datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl',
            root / 'baseline_predictions.jsonl', root / 'baseline_evaluation.json',
            root / 'predictions.jsonl', root / 'evaluation.json']

    def test_all_twenty_rows_and_actual_paired_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'comparison'
            result = COMPARE(*self.inputs(), output)
            self.assertEqual(result['base']['passed'], 12)
            self.assertEqual(result['finetuned']['passed'], 10)
            self.assertEqual(result['changes'], {'改善': 2, '退步': 4, '均通过': 8, '均未通过': 6})
            saved = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved['rows']), 20)
            self.assertTrue(all(r['base']['response'] and r['finetuned']['response'] for r in saved['rows']))
            with self.assertRaisesRegex(ValueError, '已存在'):
                COMPARE(*self.inputs(), output)

    def test_mismatched_generation_conditions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs()
            pred = root / 'base.jsonl'
            pred.write_bytes(args[1].read_bytes())
            manifest = json.loads(args[1].with_suffix('.manifest.json').read_text(encoding='utf-8'))
            manifest['max_new_tokens'] = 512
            pred.with_suffix('.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            args[1] = pred
            with self.assertRaisesRegex(ValueError, '生成条件不一致'):
                COMPARE(*args, root / 'result')
            self.assertFalse((root / 'result').exists())
