import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pyforge.data_cleaning_example import clean_rows, generate_config, DEMO_ROWS, DEFAULT_PROMPT
from pyforge.data_cleaning import SYSTEM_PROMPT


class DataCleaningExampleTests(unittest.TestCase):
    def make_run(self, root, system=SYSTEM_PROMPT):
        run = Path(root) / 'run'
        (run / 'adapter').mkdir(parents=True)
        (run / 'run_manifest.json').write_text(json.dumps({'system_prompt': system}), encoding='utf-8')
        (run / 'adapter/adapter_model.safetensors').write_bytes(b'test fixture')
        return run

    def test_model_config_is_executed_offline_without_changing_inputs(self):
        config = {'steps': [
            {'op': 'drop_null', 'columns': ['邮箱'], 'how': 'any'},
            {'op': 'deduplicate', 'columns': ['邮箱'], 'keep': 'last'},
            {'op': 'sort', 'by': [{'column': '积分', 'ascending': False}]},
        ]}
        rows = copy.deepcopy(DEMO_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            with patch('pyforge.training.chat', return_value=json.dumps(config)) as model:
                result = clean_rows(DEFAULT_PROMPT, rows, run)
            model.assert_called_once_with(run, DEFAULT_PROMPT, max_new_tokens=256, offline=True)
        self.assertEqual(rows, DEMO_ROWS)
        self.assertEqual(result['input_count'], 5)
        self.assertEqual(result['output_count'], 3)
        self.assertEqual(result['rows'], [
            {'邮箱': 'a@example.com', '积分': 90},
            {'邮箱': 'b@example.com', '积分': 80},
            {'邮箱': 'c@example.com', '积分': 60},
        ])

    def test_invalid_answer_never_reaches_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            with patch('pyforge.training.chat', return_value='```python\nprint(1)\n```'), \
                 patch('pyforge.data_cleaning_example.execute_config') as execute:
                with self.assertRaisesRegex(ValueError, '未执行清洗'):
                    clean_rows(DEFAULT_PROMPT, DEMO_ROWS, run)
                execute.assert_not_called()

    def test_wrong_model_or_invalid_input_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, system='old Python programming prompt')
            with patch('pyforge.training.chat') as model:
                with self.assertRaisesRegex(ValueError, '旧 LeetCode'):
                    generate_config(DEFAULT_PROMPT, run)
                with self.assertRaisesRegex(ValueError, '对象数组'):
                    clean_rows(DEFAULT_PROMPT, {'邮箱': 'x'}, run)
                model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
