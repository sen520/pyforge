import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_training_data import CHECKER, examples, literal, prepare, sample_rows


class DataAuditTests(unittest.TestCase):
    def test_audit_uses_training_prompt_and_rejects_unadapted_big_array(self):
        row = {'id':'fixture', 'split':'train', 'difficulty':'easy', 'problem_id':0,
               'source_url':'fixture', 'source_revision':'fixture',
               'messages':[{}, {'content':'接口：\n```python\nclass Solution:\n    def f(self, x): ...\n```'}, {'content':'answer'}]}
        question = {'title':'fixture', 'interface':'class WrongHelper: pass', 'description':'输入：x = 1\n输出：1'}
        self.assertEqual(prepare(row, question)['status'], 'ready')
        row['messages'][1]['content'] = row['messages'][1]['content'].replace('x):', "x: 'BigArray'):")
        self.assertEqual(prepare(row, question)['status'], 'unsupported_adapter')

    def test_fixed_strata_and_reject_test_split(self):
        rows = [{'id': str(i), 'split': 'train' if i < 80 else 'validation',
                 'difficulty': 'easy' if i % 2 else 'hard'} for i in range(100)]
        picked = sample_rows(rows, 20)
        self.assertEqual(picked, sample_rows(list(reversed(rows)), 20))
        self.assertEqual(len(picked), 20)
        self.assertEqual(sum(r['split']=='validation' for r in picked), 4)
        with self.assertRaises(ValueError):
            sample_rows([{'split': 'test'}], 1)

    def test_literal_parser_never_runs_calls(self):
        self.assertEqual(literal('[true, false, null]'), [True, False, None])
        for value in ('__import__("os").getcwd()', '[x for x in [1]]', 'unknown'):
            with self.assertRaises(ValueError):
                literal(value)

    def test_parse_named_examples_and_float(self):
        text = '输入：nums = [1, 2], k = 2\n输出：2.50000\n解释：example\n示例 2：\n输入：nums = [], k = 1\n输出：false\n提示：限制'
        cases = examples(text, ['nums', 'k'])
        self.assertEqual([c['expected'] for c in cases], [2.5, False])
        self.assertEqual(cases[0]['args'], [[1,2],2])
        self.assertEqual(examples('输入：nums = __import__("os")\n输出：1', ['nums']), [])

    def test_properties_accept_multiple_answers_and_reject_mutants(self):
        # 只执行项目自己的纯检查器；候选题解始终在 Docker 中执行。
        ns = {}
        exec(CHECKER, ns)
        match = ns['match']
        self.assertTrue(match(1, None, 'peak_index', [[1,3,2]]))
        self.assertFalse(match(0, None, 'peak_index', [[1,3,2]]))
        self.assertTrue(match([2,9], None, 'no_zero_pair', [11]))
        self.assertFalse(match([1,10], None, 'no_zero_pair', [11]))
        self.assertTrue(match([[0,0],[0,2],[0,1]], None, 'distance_grid', [1,3,0,0]) is False)
        self.assertTrue(match([[0,1],[0,2],[0,0]], None, 'distance_grid', [1,3,0,1]))
        self.assertTrue(match([[1,2]], None, 'smallest_pairs', [[1,2],[1,2],2]) is False)
        self.assertTrue(match([[2,1],[1,1]], None, 'smallest_pairs', [[1,2],[1,2],2]))
        self.assertFalse(match([[2,2],[1,1]], None, 'smallest_pairs', [[1,2],[1,2],2]))
        self.assertFalse(match(True, 1, 'exact', []))
        self.assertFalse(match(1, True, 'exact', []))
        self.assertTrue(match(2, 2.0, 'exact', []))
        self.assertFalse(match(['a','a'], ['a'], 'unordered', []))


if __name__ == '__main__':
    unittest.main()
