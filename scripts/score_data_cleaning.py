"""对原始 JSON 配置预测计分；不修复回答、不执行任意代码。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pyforge.data import read_records
from pyforge.data_cleaning import parse_config


def score(data, predictions, report, split='test'):
    if split not in ('test', 'validation'):
        raise ValueError('只允许对 test 或 validation 计分')
    report = Path(report)
    if report.exists():
        raise ValueError('报告已存在，请使用新文件名')
    reference = Path(data) / (split + '.jsonl')
    records = read_records(reference)
    predictions = Path(predictions)
    answers = {}
    for number, line in enumerate(predictions.read_text(encoding='utf-8-sig').splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not isinstance(row.get('response'), str):
            raise ValueError(f'预测第 {number} 行缺少字符串 id/response')
        if row['id'] in answers:
            raise ValueError('预测 ID 重复：' + row['id'])
        answers[row['id']] = row['response']
    unknown = set(answers) - {r['id'] for r in records}
    if unknown:
        raise ValueError('预测含未知 ID：' + ', '.join(sorted(unknown)[:5]))
    results = []
    for row in records:
        response = answers.get(row['id'])
        detail = {'id': row['id'], 'status': 'missing', 'schema_valid': False}
        if response is not None:
            try:
                config = parse_config(response)
                expected = parse_config(row['messages'][-1]['content'])
                # JSON 字典的 Python 相等会把 true 与 1 混同，用规范化 JSON 区分。
                canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                detail.update(schema_valid=True, status='correct' if canonical(config) == canonical(expected) else 'wrong_config')
            except (ValueError, TypeError) as exc:
                detail.update(status='invalid_json_or_schema', error=str(exc))
        results.append(detail)
    correct = sum(r['status'] == 'correct' for r in results)
    valid = sum(r['schema_valid'] for r in results)
    output = {
        'split': split, 'total': len(records), 'provided': len(answers),
        'correct': correct, 'exact_config_accuracy': correct / len(records),
        'schema_valid_rate': valid / len(records),
        'metric_definition': '严格参考配置匹配；忽略对象键顺序与空白，保留数组顺序和 JSON 类型。不是执行结果正确率。缺答计入分母。',
        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
        'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'results': results,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/data_cleaning')
    parser.add_argument('--split', choices=('test', 'validation'), default='test')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.data, args.predictions, args.report, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
