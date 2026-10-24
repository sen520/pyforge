"""复跑主代理独立编写的有限配置执行用例，不据此宣称全量语义通过。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pyforge.data_cleaning import execute_config


def check(directory):
    directory = Path(directory)
    source = directory / 'metadata/execution_cases.json'
    cases = json.loads(source.read_text(encoding='utf-8'))['cases']
    authored = {}
    for split in {c['split'] for c in cases}:
        path = directory / 'sources' / (split + '_authored.jsonl')
        authored[split] = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    results = []
    for case in cases:
        row = authored[case['split']][case['line'] - 1]
        fingerprint = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if case.get('authored_row_sha256') != fingerprint:
            raise ValueError(f'{case["split"]}:{case["line"]} 原稿改变，须先重新审查用例适用性')
        actual = [row['_row'] for row in execute_config(row['answer'], case['rows'])]
        results.append({'split': case['split'], 'source_line': case['line'],
                        'authored_row_sha256': fingerprint, 'purpose': case['purpose'],
                        'expected_row_ids': case['expected_row_ids'], 'actual_row_ids': actual,
                        'passed': actual == case['expected_row_ids']})
    report = {'case_count': len(results), 'passed': sum(r['passed'] for r in results),
              'human_review': False, 'author': 'Codex',
              'scope': '仅列出的参考配置与独立用例，不是全部样本执行验证，也不是模型评估',
              'cases_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'results': results}
    (directory / 'metadata/execution_review.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    if report['passed'] != report['case_count']:
        raise ValueError('部分独立执行用例未通过，详见 execution_review.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/data_cleaning')
    args = parser.parse_args()
    result = check(args.data)
    print(json.dumps({'case_count': result['case_count'], 'passed': result['passed']}, ensure_ascii=False))
