"""保留原始执行结果，另存审计分类和题面接口缺失清单。"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def interface_findings():
    findings = []
    for split in ('train', 'validation'):
        for line in (ROOT / 'datasets/leetcode' / (split + '.jsonl')).read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            if row['source'] != 'doocs_reference_sft_v1':
                continue
            prompt = re.search(r'接口：\n```python\n(.*?)```', row['messages'][1]['content'], re.S)
            answer = re.search(r'```python\n(.*?)```', row['messages'][-1]['content'], re.S)
            if not prompt or not answer:
                continue
            actual = [n for n in ast.parse(answer[1]).body if isinstance(n, ast.ClassDef) and n.name == 'Solution']
            declared = [n.name for n in ast.parse(prompt[1]).body if isinstance(n, ast.ClassDef)]
            if actual and 'Solution' not in declared:
                findings.append({'id': row['id'], 'split': split, 'source_url': row['source_url'],
                    'category': 'prompt_omits_solution_interface', 'declared_classes': declared,
                    'answer_sha256': hashlib.sha256(row['messages'][-1]['content'].encode()).hexdigest(),
                    'verification': 'automatic_ast_check_not_human_review'})
    return findings


def review(output):
    sample = json.loads((output / 'sample.json').read_text(encoding='utf8'))
    results = {r['id']: r for r in map(json.loads, (output / 'results.jsonl').read_text(encoding='utf8').splitlines())}
    if len(results) != sample['sample_size']:
        raise ValueError('审计尚未完成')
    # BigArray 使用 .size()/.at()，普通数组不是其合法适配器。保留原始 traceback。
    reviewed = []
    for item in sample['items']:
        row = results[item['id']]
        category = row['status']
        reason = row.get('reason', '')
        if row['id'] == 'doocs-leetcode-2936' and category == 'runtime_error' and "has no attribute 'size'" in row.get('stderr', ''):
            category, reason = 'unsupported_adapter', 'BigArray 对象未适配，不能归因于答案错误'
        elif category == 'wrong_answer':
            category, reason = 'requires_review', '应先核对题意、比较器和用例，再判断是否答案错误'
        reviewed.append({'id': row['id'], 'raw_status': row['status'], 'category': category, 'reason': reason})
    findings = interface_findings()
    report = {'sample_sha256': hashlib.sha256((output / 'sample.json').read_bytes()).hexdigest(),
              'raw_results_sha256': hashlib.sha256((output / 'results.jsonl').read_bytes()).hexdigest(),
              'classification': 'Codex 检查题面与接口后分类，不标记人工审核；原始结果保持不变',
              'counts': dict(Counter(r['category'] for r in reviewed)), 'results': reviewed,
              'interface_findings_scope': '仅训练和验证集的全部上游答案；不读取测试集',
              'interface_findings': findings}
    (output / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'counts': report['counts'], 'interface_findings':len(findings)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/data-audit-v1')
    review(parser.parse_args().output)
