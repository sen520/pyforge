"""封装模型逐条编写的原稿；不生成、扩写或自动改写需求与答案。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pyforge.data import read_records
from pyforge.data_cleaning import SYSTEM_PROMPT, validate_config
from pyforge.encoding import encode_record
from pyforge.training_config import load_splits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_prompt(prompt):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', prompt).casefold())


def load_authored(directory):
    result, prompts, groups = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        path = directory / 'sources' / (split + '_authored.jsonl')
        rows = []
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            row = json.loads(line)
            if set(row) != {'prompt', 'answer', 'group'}:
                raise ValueError(f'{path}:{number}: 原稿字段必须为 prompt/answer/group')
            if not isinstance(row['prompt'], str) or not row['prompt'].strip():
                raise ValueError(f'{path}:{number}: 题面为空')
            if not isinstance(row['group'], str) or not row['group'].startswith(split + '_'):
                raise ValueError(f'{path}:{number}: 场景族须以 {split}_ 开头')
            validate_config(row['answer'])
            fingerprint = normalized_prompt(row['prompt'])
            if fingerprint in prompts:
                raise ValueError(f'{split}:{number} 与 {prompts[fingerprint]} 的规范化题面重复')
            prompts[fingerprint] = f'{split}:{number}'
            previous = groups.setdefault(row['group'], split)
            if previous != split:
                raise ValueError('场景族跨集合泄漏：' + row['group'])
            # 只检查显式列名覆盖，不用代码猜测中文需求的正确答案。
            for step in row['answer']['steps']:
                columns = (step['columns'] if 'columns' in step else
                           [v['column'] for v in step['by']] if 'by' in step else [step['column']])
                if any(c not in row['prompt'] for c in columns):
                    raise ValueError(f'{split}:{number}: 答案列名未在题面中出现')
            rows.append(row)
        if len(rows) < 1000:
            raise ValueError(f'{split} 只有 {len(rows)} 条；本批要求每个集合至少 1000 条')
        result[split] = rows
    return result


def build(directory, tokenizer=None, max_length=2048, check=False):
    directory = Path(directory)
    authored = load_authored(directory)
    outputs = {}
    for split, rows in authored.items():
        records = []
        for number, row in enumerate(rows, 1):
            answer = json.dumps(row['answer'], ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            record = {
                'id': f'data-cleaning-v1-{split}-{number:04d}',
                'task_type': 'data_cleaning_config', 'language': 'zh',
                'source': 'codex_direct_authored_data_cleaning_v1',
                'license': 'project_authored_license_unspecified',
                'author': 'Codex', 'group_id': row['group'],
                'source_path': f'datasets/data_cleaning/sources/{split}_authored.jsonl',
                'source_line': number,
                'verification': {'schema': True, 'execution': False, 'human_review': False},
                'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                             {'role': 'user', 'content': row['prompt']},
                             {'role': 'assistant', 'content': answer}],
            }
            if tokenizer is not None:
                encoded = encode_record(record, tokenizer, max_length)
                if encoded is None:
                    raise ValueError(f'{record["id"]}: 超过 {max_length} token，不静默过滤')
                record['token_length'] = len(encoded['input_ids'])
            records.append(record)
        outputs[split] = records

    report = {
        'version': 'data_cleaning_v1', 'author': 'Codex',
        'authoring_method': '模型在当前对话及获用户授权的三个子代理中逐条编写需求与答案；代码仅序列化与检查',
        'external_model_api_calls': 0, 'minimum_per_split': 1000,
        'semantic_review': '作者逐条编写；自动格式/重复/列名/长度检查不能证明每条中文语义正确；抽查另见 semantic_review.json',
        'human_review': False, 'per_record_execution': False,
        'split_policy': '三个集合预先分配不同业务领域与场景族；未按条随机切分；不宣称已完成所有语义近似去重',
        'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        'schema_contract_sha256': digest(directory / '配置约定.md'),
        'builder_sha256': digest(__file__),
        'validator_sha256': digest(ROOT / 'src/pyforge/data_cleaning.py'),
        'tokenizer_checked': tokenizer is not None,
        'tokenizer_snapshot': Path(tokenizer.name_or_path).name if tokenizer is not None else None,
        'tokenizer_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() if tokenizer is not None else None,
        'max_length': max_length if tokenizer is not None else None,
        'files': {},
    }
    for split, records in outputs.items():
        path = directory / (split + '.jsonl')
        content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in records)
        if check:
            if path.read_bytes() != content.encode():
                raise ValueError(str(path) + ' 与当前原稿或封装参数不一致')
        else:
            path.write_text(content, encoding='utf-8', newline='\n')
        read_records(path)
        operation_counts = Counter(s['op'] for r in records for s in json.loads(r['messages'][-1]['content'])['steps'])
        report['files'][split] = {
            'count': len(records), 'sha256': digest(path),
            'authored_source_sha256': digest(directory / 'sources' / (split + '_authored.jsonl')),
            'groups': len({r['group_id'] for r in records}),
            'operation_counts': dict(operation_counts),
            'step_counts': dict(Counter(str(len(json.loads(r['messages'][-1]['content'])['steps'])) for r in records)),
            'max_token_length': max((r.get('token_length', 0) for r in records), default=0) or None,
        }
    load_splits(directory)
    report['checks'] = {'record_format': True, 'answer_schema': True,
                        'normalized_prompt_unique_across_splits': True,
                        'id_group_and_exact_conversation_disjoint': True,
                        'answer_columns_explicit_in_prompt': True}
    manifest = directory / 'metadata' / 'manifest.json'
    if check:
        if json.loads(manifest.read_text(encoding='utf-8')) != report:
            raise ValueError('现有 manifest 与原稿、检查脚本或 tokenizer 不一致')
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/data_cleaning')
    parser.add_argument('--tokenizer', type=Path, help='本地 tokenizer 快照路径，严格离线读取')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--check', action='store_true', help='核对现有文件，不写入')
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    result = build(args.data, tokenizer, args.max_length, args.check)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
