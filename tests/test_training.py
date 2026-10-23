import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyforge.training_config import check_resume, load_config, load_splits


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_offline_setting_applies_to_peft_save_and_hub_cache(self):
        from pyforge.paths import MODEL_CACHE_DIR
        from pyforge.training import configure_hub_runtime
        with patch.dict(os.environ, {}, clear=True):
            configure_hub_runtime(True)
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['HF_HUB_CACHE'], str(MODEL_CACHE_DIR))

    def test_reject_invalid_configuration_before_loading_model(self):
        raw = json.loads((ROOT / 'configs/smoke_4060.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            for change in ({'learning_rate': float('nan')}, {'epochs': float('inf')}, {'max_length': True},
                           {'target_modules': []}, {'save_steps': 3, 'eval_steps': 2}, {'typo': 1}):
                path.write_text(json.dumps({**raw, **change}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_resume_requires_complete_and_matching_checkpoint(self):
        config = load_config(ROOT / 'configs/smoke_4060.json')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = output / 'checkpoint-2'
            checkpoint.mkdir()
            hashes = {'train': 'a', 'validation': 'b', 'test': 'c'}
            (output / 'run_manifest.json').write_text(json.dumps({'config': config, 'data_sha256': hashes}))
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json', 'adapter_model.safetensors'):
                (checkpoint / name).touch()
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 2, 'max_steps': 10}))
            check_resume(checkpoint, output, config, hashes)
            with self.assertRaises(ValueError):
                check_resume(checkpoint, output, config, {**hashes, 'train': 'changed'})
            with self.assertRaises(ValueError):
                check_resume(checkpoint, output, {**config, 'seed': 43}, hashes)
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 10, 'max_steps': 10}))
            with self.assertRaisesRegex(ValueError, '总步数'):
                check_resume(checkpoint, output, config, hashes)
            (checkpoint / 'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                check_resume(checkpoint, output, config, hashes)

    def test_dataset_group_leakage_rejected(self):
        rows, _ = load_splits(ROOT / 'datasets')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows['validation'][0]['group_id'] = rows['train'][0]['group_id']
            for split, records in rows.items():
                (root / (split + '.jsonl')).write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'group_id'):
                load_splits(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), '需要可选 PyTorch 依赖')
    def test_collator_preserves_real_eos_and_masks_only_padding(self):
        from pyforge.encoding import CausalCollator
        result = CausalCollator(3)([
            {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3], 'attention_mask': [1, 1, 1]},
            {'input_ids': [1, 3], 'labels': [-100, 3], 'attention_mask': [1, 1]},
        ])
        self.assertEqual(result['labels'].tolist(), [[-100, 2, 3], [-100, 3, -100]])
        self.assertEqual(result['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])


if __name__ == '__main__':
    unittest.main()
