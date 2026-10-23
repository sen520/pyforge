"""按同题 ID 比较基础/微调预测与执行报告，生成逐题表格和原始答案对照。"""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path


LABELS = {'passed': '通过', 'wrong_answer': '答案错误', 'runtime_error': '运行错误',
    'timeout': '超时', 'extraction_error': '代码提取失败', 'syntax_error': '语法错误',
    'missing_prediction': '缺少回答', 'invalid_output': '输出格式错误', 'output_limit': '输出超限'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, expected, label):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError(label + ' 存在重复、缺失或未知题目 ID')
    return result


def compare(suite, base_predictions, base_report, fine_predictions, fine_report, output):
    paths = dict(suite=Path(suite), base_predictions=Path(base_predictions), base_report=Path(base_report),
        finetuned_predictions=Path(fine_predictions), finetuned_report=Path(fine_report))
    output = Path(output)
    if output.exists():
        raise ValueError('输出目录已存在，请使用新版本目录')
    problems = [json.loads(l) for l in paths['suite'].read_text(encoding='utf-8').splitlines()]
    ids = {p['id'] for p in problems}
    indexed(problems, ids, '题库')
    models = {}
    for mode, pred_key, report_key in (('base', 'base_predictions', 'base_report'),
                                     ('finetuned', 'finetuned_predictions', 'finetuned_report')):
        pred_path = paths[pred_key]
        manifest = json.loads(pred_path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        report = json.loads(paths[report_key].read_text(encoding='utf-8'))
        if manifest['status'] != 'completed' or manifest['mode'] != mode:
            raise ValueError('预测未完成或模型角色不符：' + mode)
        if report['suite_sha256'] != sha(paths['suite']) or manifest['suite_sha256'] != sha(paths['suite']):
            raise ValueError('题库哈希不一致')
        if report['predictions_sha256'] != sha(pred_path) or manifest['predictions_sha256'] != sha(pred_path):
            raise ValueError('预测哈希不一致')
        predictions = indexed([json.loads(l) for l in pred_path.read_text(encoding='utf-8').splitlines()], ids, mode + ' 预测')
        results = indexed(report['results'], ids, mode + ' 执行结果')
        if report['total'] != len(ids) or report['passed'] != sum(r['status'] == 'passed' for r in results.values()):
            raise ValueError('执行报告汇总与逐题结果不一致')
        if manifest['count'] != len(ids):
            raise ValueError('预测清单数量与题库不符')
        models[mode] = dict(manifest=manifest, report=report, predictions=predictions, results=results)
    for field in ('model_revision', 'suite_sha256', 'run_manifest_sha256', 'max_new_tokens', 'do_sample'):
        if models['base']['manifest'][field] != models['finetuned']['manifest'][field]:
            raise ValueError('生成条件不一致：' + field)
    for field in ('image_id', 'timeout_seconds_per_problem', 'execution_mode'):
        if models['base']['report'][field] != models['finetuned']['report'][field]:
            raise ValueError('执行条件不一致：' + field)
    rows = []
    for p in problems:
        values = {}
        for mode in models:
            prediction = models[mode]['predictions'][p['id']]
            result = models[mode]['results'][p['id']]
            values[mode] = {**prediction, **result, 'status_label': LABELS.get(result['status'], result['status'])}
        bp, fp = (values[m]['status'] == 'passed' for m in ('base', 'finetuned'))
        change = '改善' if fp and not bp else '退步' if bp and not fp else '均通过' if bp else '均未通过'
        rows.append({'id': p['id'], 'title': p['title'], 'difficulty': p['difficulty'], 'prompt': p['prompt'],
            'problem_url': p['problem_url'], 'test_count': len(p['tests']), 'base': values['base'],
            'finetuned': values['finetuned'], 'change': change})
    summary = {mode: {'passed': models[mode]['report']['passed'], 'total': len(ids),
        'pass_at_1': models[mode]['report']['passed'] / len(ids)} for mode in models}
    summary['changes'] = dict(Counter(r['change'] for r in rows))
    summary['percentage_point_change'] = 100 * (summary['finetuned']['pass_at_1'] - summary['base']['pass_at_1'])
    result = {'summary': summary, 'rows': rows, 'sources': {k: {'path': p.as_posix(), 'sha256': sha(p)} for k, p in paths.items()},
        'conditions': {field: models['base']['manifest'][field] for field in ('model_revision', 'max_new_tokens', 'do_sample')},
        'image_id': models['base']['report']['image_id'],
        'timeout_seconds': models['base']['report']['timeout_seconds_per_problem'], 'builder_sha256': sha(__file__)}
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    lines = ['# 基础模型与微调模型逐题对比', '',
        f"基础模型 {summary['base']['passed']}/{len(ids)}，微调模型 {summary['finetuned']['passed']}/{len(ids)}，通过率变化 {summary['percentage_point_change']:+.1f} 个百分点。", '',
        '相同题面、system 提示、tokenizer、模型快照、贪心生成长度、Docker 镜像及超时。每题单次生成，全部本地用例通过才计为通过。', '',
        '| 题号 | 题目 | 基础模型 | 微调模型 | 变化 |', '| --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['id']} | {row['title'].replace('|', '/')} | {row['base']['status_label']} | {row['finetuned']['status_label']} | {row['change']} |")
    lines += ['', '改善与退步只表示同题执行通过状态变化；样本仅 20 题，不代表整体编程能力或官方隐藏测试成绩。', '',
        '逐题完整题面和两边原始回答见同目录的《逐题答案对比.html》，机器可读数据见 comparison.json。']
    (output / '逐题对比.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    esc = html.escape
    sections = []
    for row in rows:
        sections.append(f"<details><summary>{esc(row['id'])} {esc(row['title'])}：{esc(row['base']['status_label'])} / {esc(row['finetuned']['status_label'])}（{row['change']}）</summary><pre>{esc(row['prompt'])}</pre><div class='answers'><section><h3>基础模型原始回答</h3><pre>{esc(row['base']['response'])}</pre></section><section><h3>微调模型原始回答</h3><pre>{esc(row['finetuned']['response'])}</pre></section></div></details>")
    table = '<table><tr><th>题号</th><th>题目</th><th>基础模型</th><th>微调模型</th><th>变化</th></tr>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(str(v)) + '</td>' for v in (r['id'], r['title'], r['base']['status_label'], r['finetuned']['status_label'], r['change'])) + '</tr>' for r in rows) + '</table>'
    page = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>模型评估对比</title><style>body{font:16px system-ui;margin:32px;color:#172033}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2fa}details{margin:18px 0;border:1px solid #ddd;padding:12px}summary{cursor:pointer;font-weight:bold}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:14px;font-size:13px}@media(max-width:800px){.answers{grid-template-columns:1fr}}</style><h1>20 道算法题模型对比</h1>"
    page += f"<p>基础模型 {summary['base']['passed']}/20；微调模型 {summary['finetuned']['passed']}/20。相同生成与执行条件，点击下方题目查看原始答案。</p>" + table + ''.join(sections) + '</html>'
    (output / '逐题答案对比.html').write_text(page, encoding='utf-8', newline='\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='datasets/benchmarks/doocs-leetcode/executable_v1.jsonl')
    parser.add_argument('--base-predictions', required=True)
    parser.add_argument('--base-report', required=True)
    parser.add_argument('--finetuned-predictions', required=True)
    parser.add_argument('--finetuned-report', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    print(json.dumps(compare(a.suite, a.base_predictions, a.base_report, a.finetuned_predictions,
        a.finetuned_report, a.output), ensure_ascii=False, indent=2))
