import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pyforge.benchmark import benchmark, generate


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'reports/train-doocs-001'


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'outputs').mkdir(exist_ok=True)

    def run_directory(self, root):
        run = root / 'run'
        (run / 'adapter').mkdir(parents=True)
        shutil.copyfile(FIXTURES / 'run_manifest.json', run / 'run_manifest.json')
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer_config.json'):
            (run / 'adapter' / name).touch()
        return run

    def test_complete_workflow_pairs_all_twenty_and_outputs_table(self):
        image = json.loads((FIXTURES / 'evaluation.json').read_text(encoding='utf-8'))['image_id']
        calls = []
        def prediction(run, suite, target, data, length, base):
            calls.append((base, length))
            prefix = 'baseline_predictions' if base else 'predictions'
            shutil.copyfile(FIXTURES / (prefix + '.jsonl'), target)
            shutil.copyfile(FIXTURES / (prefix + '.manifest.json'), target.with_suffix('.manifest.json'))
        def evaluation(suite, predictions, report, used_image, timeout):
            self.assertEqual(used_image, image)
            self.assertEqual(timeout, 10)
            name = 'baseline_evaluation.json' if predictions.name.startswith('base_') else 'evaluation.json'
            shutil.copyfile(FIXTURES / name, report)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as tmp:
            root = Path(tmp)
            run = self.run_directory(root)
            with patch('pyforge.benchmark.docker_environment', return_value=image), \
                 patch('pyforge.benchmark.generate', side_effect=prediction), \
                 patch('pyforge.benchmark.evaluate', side_effect=evaluation):
                result = benchmark(run, root / 'output', data=ROOT / 'datasets')
            self.assertEqual(calls, [(True, 1024), (False, 1024)])
            self.assertEqual(result['summary']['base']['passed'], 12)
            self.assertEqual(result['summary']['finetuned']['passed'], 10)
            self.assertTrue(Path(result['table']).is_file())
            self.assertTrue(Path(result['answers_table']).is_file())
            self.assertEqual(json.loads((root / 'output/pipeline_status.json').read_text(encoding='utf-8'))['status'], 'completed')

    def test_missing_docker_fails_before_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as tmp:
            root = Path(tmp)
            run = self.run_directory(root)
            with patch('pyforge.benchmark.docker_environment', side_effect=RuntimeError('Docker unavailable')), \
                 patch('pyforge.benchmark.generate') as prediction:
                with self.assertRaisesRegex(RuntimeError, 'Docker'):
                    benchmark(run, root / 'output', data=ROOT / 'datasets')
                prediction.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_generation_failure_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as tmp:
            root = Path(tmp)
            run = self.run_directory(root)
            with patch('pyforge.benchmark.docker_environment', return_value='sha256:fake'), \
                 patch('pyforge.benchmark.generate', side_effect=RuntimeError('generation failed')):
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    benchmark(run, root / 'output', data=ROOT / 'datasets')
            status = json.loads((root / 'output/pipeline_status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['stage'], 'generating_base')
            self.assertFalse((root / 'output/comparison').exists())

    def test_subprocess_arguments_share_parameters_and_select_base(self):
        for base in (True, False):
            with patch('pyforge.benchmark.subprocess.run') as process:
                process.return_value.returncode = 0
                generate('run', 'suite', 'out', 'data', 1024, base)
                command = process.call_args.args[0]
                self.assertIn('pyforge.prediction', command)
                self.assertEqual('--base' in command, base)
                self.assertEqual(command[command.index('--max-new-tokens') + 1], '1024')
