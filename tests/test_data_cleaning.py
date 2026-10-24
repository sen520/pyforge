import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from pyforge.data_cleaning import execute_config, parse_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class DataCleaningTests(unittest.TestCase):
    def test_first_last_preserve_surviving_row_order(self):
        rows = [{'id': 'x', 'v': 1}, {'id': 'y', 'v': 2}, {'id': 'x', 'v': 3}]
        for keep, expected in [('first', [1, 2]), ('last', [2, 3])]:
            config = {'steps': [{'op': 'deduplicate', 'columns': ['id'], 'keep': keep}]}
            self.assertEqual([r['v'] for r in execute_config(config, rows)], expected)
        self.assertEqual(rows[0]['v'], 1)

    def test_composite_dedup_and_strict_boolean_equality(self):
        config = {'steps': [{'op': 'deduplicate', 'columns': ['x', 'y'], 'keep': 'first'}]}
        rows = [{'x': True, 'y': 0}, {'x': 1, 'y': 0}, {'x': 1.0, 'y': 0},
                {'x': 1, 'y': None}, {'x': True, 'y': 0}]
        self.assertEqual(execute_config(config, rows), rows[:2] + rows[3:4])

    def test_null_any_all_do_not_treat_false_zero_or_blank_as_null(self):
        rows = [{'a': None, 'b': None}, {'a': None, 'b': 0}, {'a': '', 'b': False}]
        for how, expected in [('any', rows[2:]), ('all', rows[1:])]:
            config = {'steps': [{'op': 'drop_null', 'columns': ['a', 'b'], 'how': how}]}
            self.assertEqual(execute_config(config, rows), expected)

    def test_numeric_comparison_boundaries(self):
        rows = [{'n': 17}, {'n': 18}, {'n': 19}, {'n': None}]
        for op, values in [('gt', [19]), ('ge', [18, 19]), ('lt', [17]), ('le', [17, 18]),
                           ('eq', [18]), ('ne', [17, 19]), ('in', [18]), ('not_in', [17, 19])]:
            value = [18] if op in ('in', 'not_in') else 18
            config = {'steps': [{'op': 'filter', 'column': 'n', 'operator': op, 'value': value}]}
            self.assertEqual([r['n'] for r in execute_config(config, rows)], values)

    def test_string_boolean_and_number_distinction(self):
        rows = [{'x': 1}, {'x': True}, {'x': '1'}, {'x': None}]
        config = {'steps': [{'op': 'filter', 'column': 'x', 'operator': 'eq', 'value': True}]}
        self.assertEqual(execute_config(config, rows), [{'x': True}])
        config['steps'][0].update(operator='not_in', value=[1, True])
        self.assertEqual(execute_config(config, rows), [{'x': '1'}])

    def test_multikey_stable_sort_with_null_last_even_descending(self):
        rows = [{'a': None, 'b': 9, 'id': 'null'}, {'a': 1, 'b': 3, 'id': 'first'},
                {'a': 2, 'b': 1, 'id': 'high'}, {'a': 1, 'b': 3, 'id': 'second'},
                {'a': 1, 'b': 2, 'id': 'low'}]
        config = {'steps': [{'op': 'sort', 'by': [{'column': 'a', 'ascending': False},
                                                {'column': 'b', 'ascending': True}]}]}
        self.assertEqual([r['id'] for r in execute_config(config, rows)],
                         ['high', 'low', 'first', 'second', 'null'])

    def test_pipeline_order_changes_the_result(self):
        rows = [{'id': 'a', 'score': 1}, {'id': 'a', 'score': 9}, {'id': 'b', 'score': 5}]
        dedup = {'op': 'deduplicate', 'columns': ['id'], 'keep': 'first'}
        sort = {'op': 'sort', 'by': [{'column': 'score', 'ascending': False}]}
        self.assertEqual(execute_config({'steps': [dedup, sort]}, rows), [rows[2], rows[0]])
        self.assertEqual(execute_config({'steps': [sort, dedup]}, rows), [rows[1], rows[2]])

    def test_reject_ambiguous_or_extra_json_and_malformed_operations(self):
        for response in ('{"steps":[],"steps":[]}', '```json\n{"steps":[]}\n```',
                         '{"steps":[{"op":"filter","column":"n","operator":"eq","value":NaN}]}'):
            with self.assertRaises(ValueError):
                parse_config(response)
        good = {'steps': [{'op': 'drop_null', 'columns': ['a'], 'how': 'any'}]}
        bads = [None, {}, {'steps': []}, {'steps': [None]}, {'steps': [{'op': 'eval', 'code': 'x'}]}]
        for mutate in ({'columns': ['a', 'a']}, {'how': 'sometimes'}, {'extra': True}):
            changed = copy.deepcopy(good)
            changed['steps'][0].update(mutate)
            bads.append(changed)
        bads += [{'steps': [{'op': 'filter', 'column': 'n', 'operator': 'in', 'value': []}]},
                 {'steps': [{'op': 'sort', 'by': [{'column': 'x', 'ascending': 'false'}]}]}]
        for config in bads:
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_schema_and_executor_reject_missing_or_incompatible_values(self):
        config = {'steps': [{'op': 'filter', 'column': 'n', 'operator': 'gt', 'value': 10}]}
        for rows in ([{'n': '12'}], [{'other': 12}]):
            with self.assertRaises(ValueError):
                execute_config(config, rows)

    def test_builder_refuses_underfilled_sources(self):
        builder = runpy.run_path(str(ROOT / 'scripts/build_data_cleaning.py'))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'sources').mkdir()
            row = {'prompt': '按 `a` 去重保留首条。', 'group': 'train_example',
                   'answer': {'steps': [{'op': 'deduplicate', 'columns': ['a'], 'keep': 'first'}]}}
            (directory / 'sources/train_authored.jsonl').write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '至少 1000'):
                builder['load_authored'](directory)
            self.assertFalse((directory / 'train.jsonl').exists())

    def test_holdout_selection_covers_rare_operations_and_preserves_records(self):
        select = runpy.run_path(str(ROOT / 'scripts/build_data_cleaning.py'))['select_holdout']
        rows = []
        for group in ('weather', 'sports'):
            for number in range(8):
                step = ({'op': 'filter', 'column': 'n', 'operator': 'gt', 'value': 0}
                        if number < 7 else {'op': 'sort', 'by': [{'column': 'n', 'ascending': True}]})
                rows.append({'id': f'{group}-{number}', 'group_id': group,
                             'messages': [{'role': 'assistant', 'content': json.dumps({'steps': [step]})}]})
        before = copy.deepcopy(rows)
        selected = select(rows, target=4)
        self.assertEqual(selected, select(rows, target=4))
        self.assertEqual(rows, before)
        self.assertEqual(len(selected), 4)
        for group in ('weather', 'sports'):
            members = [r for r in selected if r['group_id'] == group]
            self.assertEqual(len(members), 2)
            self.assertEqual({json.loads(r['messages'][-1]['content'])['steps'][0]['op']
                              for r in members}, {'filter', 'sort'})
        self.assertEqual(selected, [r for r in rows if r['id'] in {v['id'] for v in selected}])
        self.assertEqual(select(rows, target=len(rows)), rows)
        for target in (0, 1, 2, 17):
            with self.assertRaises(ValueError):
                select(rows, target)

    def test_scoring_counts_missing_and_rejects_boolean_numeric_confusion(self):
        scorer = runpy.run_path(str(ROOT / 'scripts/score_data_cleaning.py'))['score']
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = []
            for number, value in enumerate([1, 'open', 20]):
                config = {'steps': [{'op': 'filter', 'column': 'v', 'operator': 'eq', 'value': value}]}
                records.append({'id': str(number), 'task_type': 'data_cleaning_config',
                                'language': 'zh', 'source': 'unit_test', 'license': 'test',
                                'group_id': str(number), 'messages': [
                                    {'role': 'user', 'content': 'explicit test fixture ' + str(number)},
                                    {'role': 'assistant', 'content': json.dumps(config)}]})
            (directory / 'test.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            wrong_type = {'steps': [{'op': 'filter', 'column': 'v', 'operator': 'eq', 'value': True}]}
            predicted = [{'id': '0', 'response': json.dumps(wrong_type)},
                         {'id': '1', 'response': records[1]['messages'][-1]['content']}]
            path = directory / 'predictions.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in predicted), encoding='utf-8')
            result = scorer(directory, path, directory / 'report.json')
            self.assertEqual(result['total'], 3)
            self.assertEqual(result['correct'], 1)
            self.assertEqual([r['status'] for r in result['results']], ['wrong_config', 'correct', 'missing'])
            with self.assertRaisesRegex(ValueError, '报告已存在'):
                scorer(directory, path, directory / 'report.json')


if __name__ == '__main__':
    unittest.main()
