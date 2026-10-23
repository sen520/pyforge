import ast
import hashlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
with patch.object(sys,'path',[str(ROOT/'scripts'),*sys.path]):
    BUILDER=runpy.run_path(str(ROOT/'scripts/build_codex_data.py'))


class CodexDataTests(unittest.TestCase):
    def test_frozen_dataset_hashes_and_execution_status(self):
        root=ROOT/'datasets'
        metadata=root/'metadata/codex_v1'
        manifest=json.loads((metadata/'dataset_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['verified'],100)
        self.assertEqual(manifest['external_model_api_calls'],0)
        for split,digest in manifest['files_sha256'].items():
            path=(metadata if split=='pending_review' else root/'archive/codex_v1')/(split+'.jsonl')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
        checks=[json.loads(line) for line in (metadata/'verification.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(checks),100)
        self.assertTrue(all(row['status']=='passed' for row in checks))
        self.assertEqual(sum(row.get('mutant_detected',False) for row in checks),8)

    def test_complete_authorship_and_disjoint_groups(self):
        rows=BUILDER['authored_rows'](ROOT/'datasets/sources/seed_questions.jsonl')
        self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        self.assertTrue(all(r['source']=='codex_authored_v1' for r in rows))
        groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ('train','validation','test')}
        self.assertFalse(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['test']&groups['validation'])
        for row in rows:
            code=row['messages'][-1]['content'].split('```python\n')[1].split('```')[0]
            ast.parse(code)

    def test_offline_build_no_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen',side_effect=AssertionError('no network')):
            result=BUILDER['build'](ROOT/'datasets/sources/seed_questions.jsonl',Path(tmp)/'dataset')
            self.assertEqual(result['external_model_api_calls'],0)
            self.assertEqual(result['count'],100)
            with self.assertRaises(ValueError):
                BUILDER['build'](ROOT/'datasets/sources/seed_questions.jsonl',Path(tmp)/'dataset')


if __name__=='__main__':
    unittest.main()
