"""按审计清单修复训练/验证题面的主接口；不读取测试集，不改变答案或划分。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from pyforge.encoding import encode_record
from pyforge.leetcode import extract_problem, replace_interface
from pyforge.training import configure_hub_runtime, resolve_snapshot

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore(output):
    """由逐条修复清单重建审计前 train/validation，供旧模型评估使用。"""
    output = Path(output)
    if output.exists():
        raise ValueError('恢复目录已存在')
    current = json.loads((ROOT / 'datasets/metadata/current_manifest.json').read_text(encoding='utf8'))
    manifest = json.loads((ROOT / current['repair_manifest']).read_text(encoding='utf8'))
    changes = {r['id']:r for r in manifest['changes']}
    pending = {}
    for split in ('train', 'validation'):
        raw = (ROOT / 'datasets' / (split + '.jsonl')).read_bytes()
        if sha(raw) != manifest['files'][split]['after_sha256']:
            raise ValueError('当前数据不匹配修复版本')
        lines = []
        for line in raw.decode('utf8').splitlines(keepends=True):
            row = json.loads(line)
            if row['id'] in changes:
                change = changes[row['id']]
                row['messages'][1]['content'] = change['old_user_message']
                row['token_length'] = change['old_token_length']
                line = json.dumps(row, ensure_ascii=False) + ('\r\n' if line.endswith('\r\n') else '\n')
            lines.append(line)
        data = ''.join(lines).encode('utf8')
        if sha(data) != manifest['files'][split]['before_sha256']:
            raise ValueError('恢复结果哈希不匹配')
        pending[split] = data
    output.mkdir(parents=True)
    for split,data in pending.items():
        (output / (split + '.jsonl')).write_bytes(data)


def repair(output, apply=False):
    from transformers import AutoTokenizer
    output = Path(output)
    if output.exists():
        raise ValueError('输出目录已存在，换新目录避免覆盖历史')
    review = json.loads((ROOT / 'reports/data-audit-v1/review.json').read_text(encoding='utf8'))
    audit = json.loads((ROOT / 'reports/data-audit-v1/sample.json').read_text(encoding='utf8'))
    findings = {r['id']: r for r in review['interface_findings']}
    configure_hub_runtime(True)
    snapshot, revision = resolve_snapshot('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'main', True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    changes, files, pending = [], {}, {}
    for split in ('train', 'validation'):
        path = ROOT / 'datasets' / (split + '.jsonl')
        raw_bytes = path.read_bytes()
        if sha(raw_bytes) != audit['dataset_sha256'][split]:
            raise ValueError('数据不是审计时的冻结版本：' + split)
        lines = raw_bytes.decode('utf8').splitlines(keepends=True)
        new_lines = []
        for line in lines:
            row = json.loads(line)
            if row['id'] not in findings:
                new_lines.append(line)
                continue
            if sha(row['messages'][-1]['content'].encode()) != findings[row['id']]['answer_sha256']:
                raise ValueError('答案已变化：' + row['id'])
            raw = (ROOT / 'data/sources/doocs-leetcode' / row['source_path']).read_text(encoding='utf8')
            if sha(raw.encode()) != row['source_sha256']:
                raise ValueError('上游来源已变化')
            interface = extract_problem(raw)['interface']
            if not interface.startswith('class Solution:'):
                raise ValueError('未获得主接口：' + row['id'])
            before = copy.deepcopy(row)
            old = re.search(r'接口：\n```python\n(.*?)\n```', row['messages'][1]['content'], re.S)
            if not old:
                raise ValueError('缺少接口块')
            row['messages'][1]['content'] = replace_interface(row['messages'][1]['content'], old[1], interface)
            encoded = encode_record(row, tokenizer, 2048)
            if encoded is None:
                raise ValueError('修复后超出长度：' + row['id'])
            row['token_length'] = len(encoded['input_ids'])
            changes.append({'id': row['id'], 'split': split, 'source_url': row['source_url'],
                'old_interface': old[1], 'new_interface': interface,
                'old_user_message': before['messages'][1]['content'],
                'new_user_message': row['messages'][1]['content'],
                'old_token_length': before['token_length'], 'new_token_length': row['token_length'],
                'answer_sha256_unchanged': sha(row['messages'][-1]['content'].encode())})
            new_lines.append(json.dumps(row, ensure_ascii=False) + ('\r\n' if line.endswith('\r\n') else '\n'))
        data = ''.join(new_lines).encode('utf8')
        pending[path] = data
        files[split] = {'count': len(lines), 'before_sha256':sha(raw_bytes), 'after_sha256':sha(data)}
    if len(changes) != len(findings):
        raise ValueError('修复数量与审计清单不一致')
    output.mkdir(parents=True)
    manifest = {'operation': 'repair_prompt_primary_interface_only', 'applied': apply,
                'changes_count': len(changes), 'files': files, 'changes': changes,
                'tokenizer': 'Qwen/Qwen2.5-Coder-0.5B-Instruct', 'tokenizer_revision': revision,
                'max_length':2048, 'external_model_api_calls':0,
                'unchanged': ['answer_content', 'split_membership', 'source_attribution', 'test_file'],
                'verification': 'automatic_source_hash_AST_and_tokenizer_check_not_human_review'}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    if apply:
        for path, data in pending.items():
            path.write_bytes(data)
        original = json.loads((ROOT / 'datasets/metadata/doocs_sft_v1/manifest.json').read_text(encoding='utf8'))
        current = {'version': 'interface_repair_v1', 'parent': 'doocs_sft_v1',
                   'repair_manifest': output.resolve().relative_to(ROOT).as_posix() + '/manifest.json',
                   'files': copy.deepcopy(original['files'])}
        for split, info in files.items():
            current['files'][split]['sha256'] = info['after_sha256']
        (ROOT / 'datasets/metadata/current_manifest.json').write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'changes':len(changes), 'applied':apply, 'files':files}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/metadata/interface_repair_v1')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', action='store_true', help='由修复清单恢复旧 train/validation 到 --output；不修改正式集合')
    args = parser.parse_args()
    if args.restore:
        if args.apply:
            parser.error('--restore 与 --apply 不能同时使用')
        restore(args.output)
    else:
        repair(args.output, args.apply)
