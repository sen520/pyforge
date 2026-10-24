import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pyforge.data_cleaning import SYSTEM_PROMPT
from pyforge.data_cleaning_benchmark import (
    benchmark, judge, prompt_rows, sha, validate_predictions, write_json,
)


class DataCleaningBenchmarkTests(unittest.TestCase):
    def make_fixture(self, root):
        run, data = root / 'run', root / 'data'
        (run / 'adapter').mkdir(parents=True)
        data.mkdir()
        (run / 'adapter/adapter_model.safetensors').write_bytes(b'adapter fixture')
        (data / 'train.jsonl').write_text('training hash fixture', encoding='utf-8')
        expected = {'steps': [{'op': 'filter', 'column': 'age', 'operator': 'ge', 'value': 18}]}
        records = {}
        for split in ('validation', 'test'):
            records[split] = [{
                'id': f'{split}-{number}', 'group_id': split, 'task_type': 'data_cleaning_config',
                'language': 'zh', 'source': 'unit_test', 'license': 'test',
                'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                             {'role': 'user', 'content': f'{split} fixture {number}: age at least 18'},
                             {'role': 'assistant', 'content': json.dumps(expected)}],
            } for number in range(2)]
            (data / f'{split}.jsonl').write_text(
                ''.join(json.dumps(r) + '\n' for r in records[split]), encoding='utf-8')
        manifest = {'system_prompt': SYSTEM_PROMPT, 'model_revision': 'snapshot-fixture',
                    'data_sha256': {'train': sha(data / 'train.jsonl')}}
        write_json(run / 'run_manifest.json', manifest)
        write_json(run / 'run_status.json', {'status': 'completed'})
        return run, data, manifest, records

    def predictions(self, folder, split, mode, run, records):
        path = folder / f'{split}-{mode}.jsonl'
        expected = records[split][0]['messages'][-1]['content']
        wrong = json.dumps({'steps': [{'op': 'filter', 'column': 'age', 'operator': 'gt', 'value': 18}]})
        responses = [f'```json\n{expected}\n```', wrong] if mode == 'base' else [expected, expected]
        path.write_text(''.join(json.dumps({'id': r['id'], 'response': response,
                                          'generated_tokens': 30, 'truncated': False}) + '\n'
                                for r, response in zip(records[split], responses)), encoding='utf-8')
        write_json(path.with_suffix('.manifest.json'), {
            'status': 'completed', 'mode': mode, 'count': 2,
            'predictions_sha256': sha(path), 'run_manifest_sha256': sha(run / 'run_manifest.json'),
            'model_revision': 'snapshot-fixture',
            'adapter_sha256': sha(run / 'adapter/adapter_model.safetensors') if mode == 'finetuned' else None,
            'suite_sha256': sha(folder / f'{split}-prompts.jsonl'),
            'script_sha256': 'same-predict-script', 'max_new_tokens': 512, 'do_sample': False,
        })

    def make_reuse(self, root, run, records):
        directory = root / 'predictions'
        directory.mkdir()
        for split, rows in records.items():
            (directory / f'{split}-prompts.jsonl').write_text(
                ''.join(json.dumps(r) + '\n' for r in prompt_rows(rows)), encoding='utf-8')
            for mode in ('base', 'finetuned'):
                self.predictions(directory, split, mode, run, records)
        return directory

    @staticmethod
    def fake_export(output, node, modules):
        (output / '模型效果对比.xlsx').write_bytes(b'xlsx fixture; real exporter verified separately')

    def test_strict_and_diagnostic_do_not_rewrite_answers(self):
        expected = {'steps': [{'op': 'filter', 'column': 'age', 'operator': 'ge', 'value': 18}]}
        response = '```json\n' + json.dumps(expected) + '\n```'
        self.assertFalse(judge(response, expected)['schema_valid'])
        self.assertTrue(judge(response, expected, unwrap=True)['correct'])
        self.assertFalse(judge('Explanation\n' + response, expected, unwrap=True)['correct'])
        self.assertFalse(judge(response + '\n' + response, expected, unwrap=True)['correct'])
        self.assertFalse(judge(response.replace('18', 'true'), expected, unwrap=True)['correct'])

    def test_reuse_runs_no_generation_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, data, manifest, records = self.make_fixture(root)
            reuse = self.make_reuse(root, run, records)
            before = {p.name: p.read_bytes() for p in reuse.iterdir()}
            with patch('pyforge.data_cleaning_benchmark.find_runtime', return_value=(root / 'node', root / 'modules')):
                result = benchmark(run, root / 'output', data, reuse,
                                   runner=lambda *a, **k: self.fail('reuse must not generate'), exporter=self.fake_export)
            self.assertEqual(before, {p.name: p.read_bytes() for p in reuse.iterdir()})
            for split in ('validation', 'test'):
                self.assertEqual(result[split]['summary']['strict_before']['correct'], 0)
                self.assertEqual(result[split]['summary']['diagnostic_before']['correct'], 1)
                self.assertEqual(result[split]['summary']['strict_after']['correct'], 2)
                self.assertEqual(result[split]['summary']['diagnostic_changes']['改善'], 1)
            status = json.loads((root / 'output/pipeline_status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(status['excel_sha256'], sha(root / 'output/模型效果对比.xlsx'))

    def test_fresh_generation_is_serial_and_selects_base_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, data, manifest, records = self.make_fixture(root)
            calls = []
            def runner(command, **kwargs):
                path = Path(command[command.index('--output') + 1])
                split, mode = path.stem.split('-')
                self.assertEqual('--base' in command, mode == 'base')
                self.assertEqual(command[command.index('--max-new-tokens') + 1], '512')
                self.assertTrue(kwargs['check'])
                calls.append((split, mode))
                self.predictions(path.parent, split, mode, run, records)
            with patch('pyforge.data_cleaning_benchmark.find_runtime', return_value=(root / 'node', root / 'modules')):
                benchmark(run, root / 'output', data, runner=runner, exporter=self.fake_export)
            self.assertEqual(calls, [('validation', 'finetuned'), ('validation', 'base'),
                                     ('test', 'finetuned'), ('test', 'base')])

    def test_mismatched_conditions_and_tampered_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, data, manifest, records = self.make_fixture(root)
            reuse = self.make_reuse(root, run, records)
            path = reuse / 'validation-base.manifest.json'
            changed = json.loads(path.read_text(encoding='utf-8'))
            changed['max_new_tokens'] = 256
            write_json(path, changed)
            with patch('pyforge.data_cleaning_benchmark.find_runtime', return_value=(root / 'node', root / 'modules')):
                with self.assertRaisesRegex(ValueError, '生成条件不同'):
                    benchmark(run, root / 'output', data, reuse, exporter=self.fake_export)
            status = json.loads((root / 'output/pipeline_status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')
            self.assertFalse((root / 'output/模型效果对比.xlsx').exists())
            with (reuse / 'validation-base.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, '预测哈希不符'):
                validate_predictions(reuse, 'validation', 'base', records['validation'], run, manifest)

    def test_generation_failure_is_recorded_without_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, data, manifest, records = self.make_fixture(root)
            def runner(*args, **kwargs):
                raise subprocess.CalledProcessError(1, 'prediction')
            with patch('pyforge.data_cleaning_benchmark.find_runtime', return_value=(root / 'node', root / 'modules')):
                with self.assertRaises(subprocess.CalledProcessError):
                    benchmark(run, root / 'output', data, runner=runner,
                              exporter=lambda *a: self.fail('must not export failed generation'))
            status = json.loads((root / 'output/pipeline_status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['stage'], 'validation_finetuned')


if __name__ == '__main__':
    unittest.main()
