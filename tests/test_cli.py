import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from pyforge.cli import main


class CurrentCliTests(unittest.TestCase):
    def test_train_defaults_to_current_data_without_loading_model(self):
        with patch.object(sys, 'argv', ['pyforge', 'train', '--config', 'configs/train_4080.json',
                                       '--output', 'outputs/unit-test', '--offline']), \
             patch('pyforge.training.train', return_value={'completed': True}) as train, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        train.assert_called_once_with('configs/train_4080.json', 'datasets/data_cleaning',
                                      'outputs/unit-test', None, True, None)
        self.assertTrue(json.loads(output.getvalue())['completed'])

    def test_benchmark_routes_to_current_pipeline_and_reuse(self):
        results = {'test': {'summary': {'total': 200}}, 'validation': {'summary': {'total': 200}}}
        with patch.object(sys, 'argv', ['pyforge', 'benchmark', '--run', 'outputs/current-run',
                                       '--output', 'outputs/current-report',
                                       '--reuse-predictions', 'outputs/current-predictions']), \
             patch('pyforge.data_cleaning_benchmark.benchmark', return_value=results) as benchmark, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        benchmark.assert_called_once_with('outputs/current-run', 'outputs/current-report',
                                          'datasets/data_cleaning', 'outputs/current-predictions',
                                          512, None, None)
        self.assertEqual(json.loads(output.getvalue()), {'test': {'total': 200}, 'validation': {'total': 200}})

    def test_retired_commands_are_not_available(self):
        for command in ('generate', 'evaluate'):
            with patch.object(sys, 'argv', ['pyforge', command]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
