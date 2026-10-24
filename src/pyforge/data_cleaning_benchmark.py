"""顺序生成数据清洗预测、评分，并导出基础/微调模型对比工作簿。"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .data import read_records
from .data_cleaning import parse_config
from .paths import PROJECT_ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def canonical(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def judge(response, expected, unwrap=False):
    """诊断只允许去掉完整包裹回答的一个 JSON 代码块，不改写配置。"""
    wrapped = False
    text = response.strip()
    if unwrap:
        match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
        if match:
            text, wrapped = match.group(1), True
    try:
        actual = parse_config(text)
    except (ValueError, TypeError) as exc:
        return {'correct': False, 'schema_valid': False, 'wrapped': wrapped,
                'status': '格式错误', 'error': str(exc)}
    correct = canonical(actual) == canonical(expected)
    return {'correct': correct, 'schema_valid': True, 'wrapped': wrapped,
            'status': '正确' if correct else '配置不符', 'error': ''}


def change(before, after):
    if before and after:
        return '均正确'
    if not before and not after:
        return '均错误'
    return '改善' if after else '退步'


def prompt_rows(records):
    return [{'id': r['id'], 'prompt': next(m['content'] for m in r['messages'] if m['role'] == 'user')}
            for r in records]


def validate_predictions(directory, split, mode, records, run, run_manifest):
    """核对预测完整性与生成条件，拒绝把其他运行或其他题库的结果混入。"""
    path = directory / f'{split}-{mode}.jsonl'
    sidecar = path.with_suffix('.manifest.json')
    manifest = json.loads(sidecar.read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed' or manifest.get('mode') != mode:
        raise ValueError(f'{sidecar}: 预测未完成或模型类型不符')
    if manifest.get('predictions_sha256') != sha(path):
        raise ValueError(f'{path}: 预测哈希不符')
    if manifest.get('run_manifest_sha256') != sha(run / 'run_manifest.json'):
        raise ValueError(f'{path}: 预测不属于指定训练运行')
    if manifest.get('model_revision') != run_manifest['model_revision']:
        raise ValueError(f'{path}: 基础模型快照不符')
    expected_adapter = sha(run / 'adapter/adapter_model.safetensors') if mode == 'finetuned' else None
    if manifest.get('adapter_sha256') != expected_adapter:
        raise ValueError(f'{path}: 适配器哈希不符')
    suite = directory / f'{split}-prompts.jsonl'
    if manifest.get('suite_sha256') != sha(suite):
        raise ValueError(f'{path}: 生成题面哈希不符')
    original = [json.loads(line) for line in suite.read_text(encoding='utf-8-sig').splitlines()]
    if original != prompt_rows(records):
        raise ValueError(f'{suite}: 与当前 {split} 的题号、顺序或题面不一致')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    ids = [r.get('id') for r in rows]
    if (len(set(ids)) != len(ids) or set(ids) != {r['id'] for r in records} or
            any(not isinstance(r.get('response'), str) for r in rows) or
            manifest.get('count') != len(records)):
        raise ValueError(f'{path}: 题号缺失/重复/未知、回答类型错误或数量不符')
    return {r['id']: r for r in rows}, manifest


def compare_split(records, before, after):
    comparisons = []
    for record in records:
        expected = parse_config(record['messages'][-1]['content'])
        previous, current = before[record['id']], after[record['id']]
        strict_before = judge(previous['response'], expected)
        strict_after = judge(current['response'], expected)
        diagnostic_before = judge(previous['response'], expected, unwrap=True)
        diagnostic_after = judge(current['response'], expected, unwrap=True)
        comparisons.append({
            'id': record['id'], 'group': record['group_id'],
            'prompt': next(m['content'] for m in record['messages'] if m['role'] == 'user'),
            'reference': record['messages'][-1]['content'],
            'operations': ' / '.join(s['op'] for s in expected['steps']),
            'before_response': previous['response'], 'after_response': current['response'],
            'strict_before': strict_before, 'strict_after': strict_after,
            'diagnostic_before': diagnostic_before, 'diagnostic_after': diagnostic_after,
            'strict_change': change(strict_before['correct'], strict_after['correct']),
            'diagnostic_change': change(diagnostic_before['correct'], diagnostic_after['correct']),
            'before_tokens': previous.get('generated_tokens'), 'after_tokens': current.get('generated_tokens'),
            'before_truncated': previous.get('truncated'), 'after_truncated': current.get('truncated'),
        })
    summary = {'total': len(comparisons)}
    for key in ('strict_before', 'strict_after', 'diagnostic_before', 'diagnostic_after'):
        correct = sum(r[key]['correct'] for r in comparisons)
        valid = sum(r[key]['schema_valid'] for r in comparisons)
        summary[key] = {'correct': correct, 'accuracy': correct / len(comparisons),
                        'schema_valid': valid, 'schema_valid_rate': valid / len(comparisons)}
    summary['diagnostic_changes'] = {label: sum(r['diagnostic_change'] == label for r in comparisons)
                                     for label in ('改善', '退步', '均正确', '均错误')}
    return {'summary': summary, 'rows': comparisons}


def find_runtime(node=None, modules=None):
    """使用 Codex 附带运行时，也可显式指定；不安装或修改依赖。"""
    root = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
    executable = Path(node) if node else root / 'node/bin' / ('node.exe' if os.name == 'nt' else 'node')
    packages = Path(modules) if modules else root / 'node/node_modules'
    if not executable.is_file() or not (packages / '@oai/artifact-tool/package.json').is_file():
        raise ValueError('Excel 导出缺少 Codex 附带 Node/artifact-tool；请用 --node 与 --artifact-modules 指定已有运行时，不需要重新训练')
    return executable.resolve(), packages.resolve()


def export_excel(output, node, modules):
    runtime = output / '.artifact-runtime'
    runtime.mkdir()
    junction = runtime / 'node_modules'
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(modules)],
                       check=True, capture_output=True, text=True)
    else:
        junction.symlink_to(modules, target_is_directory=True)
    exporter = PROJECT_ROOT / 'scripts/export_data_cleaning_comparison.mjs'
    subprocess.run([str(node), str(exporter), '--input', str(output / 'comparison.json'),
                    '--output', str(output / '模型效果对比.xlsx'), '--runtime', str(runtime)], check=True)


def benchmark(run, output, data=None, reuse_predictions=None, max_tokens=512, node=None, modules=None,
              runner=subprocess.run, exporter=export_excel):
    run, output = Path(run).resolve(), Path(output).resolve()
    data = Path(data or PROJECT_ROOT / 'datasets/data_cleaning').resolve()
    node, modules = find_runtime(node, modules)
    if output.exists():
        raise ValueError('输出目录已存在，请使用新目录，避免混入旧预测和报告')
    if max_tokens <= 0:
        raise ValueError('max-new-tokens 必须为正数')
    run_manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    state = json.loads((run / 'run_status.json').read_text(encoding='utf-8'))
    if state.get('status') != 'completed':
        raise ValueError('训练尚未完成，run_status 必须为 completed')
    if sha(data / 'train.jsonl') != run_manifest['data_sha256']['train']:
        raise ValueError('训练数据哈希不符；请使用数据清洗运行，不能使用旧 LeetCode 模型')
    records = {split: read_records(data / f'{split}.jsonl') for split in ('test', 'validation')}
    for split, rows in records.items():
        if any(r['messages'][0].get('role') != 'system' or
               r['messages'][0]['content'] != run_manifest.get('system_prompt') for r in rows):
            raise ValueError(f'{split}: system 提示与训练运行不一致')
    output.mkdir(parents=True)
    status_path = output / 'pipeline_status.json'
    provenance = {'run': str(run), 'data': str(data), 'run_manifest_sha256': sha(run / 'run_manifest.json'),
                  'model_revision': run_manifest['model_revision'],
                  'reused_predictions': str(Path(reuse_predictions).resolve()) if reuse_predictions else None,
                  'scoring': '严格评分保持原回答；诊断只去掉完整的一个JSON代码块外壳，不修改答案内容',
                  'splits': {}}
    stage = 'preparing'
    try:
        source = Path(reuse_predictions).resolve() if reuse_predictions else output
        if not reuse_predictions:
            for split, rows in records.items():
                with (output / f'{split}-prompts.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
                    for prompt in prompt_rows(rows):
                        stream.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        results = {}
        for split, rows in records.items():
            predictions, manifests = {}, {}
            for mode in ('finetuned', 'base'):
                stage = f'{split}_{mode}'
                write_json(status_path, {'status': 'running', 'stage': stage})
                print(f'{split} / {mode}: ' + ('核对已有预测' if reuse_predictions else '生成预测'), flush=True)
                if not reuse_predictions:
                    command = [sys.executable, '-X', 'utf8', '-u', '-m', 'pyforge.prediction',
                               '--run', str(run), '--data', str(data),
                               '--suite', str(output / f'{split}-prompts.jsonl'),
                               '--output', str(output / f'{split}-{mode}.jsonl'),
                               '--max-new-tokens', str(max_tokens)]
                    if mode == 'base':
                        command.append('--base')
                    child_env = dict(os.environ)
                    child_env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + child_env.get('PYTHONPATH', '')
                    runner(command, check=True, cwd=PROJECT_ROOT, env=child_env)
                predictions[mode], manifests[mode] = validate_predictions(source, split, mode, rows, run, run_manifest)
            for key in ('model_revision', 'script_sha256', 'max_new_tokens', 'do_sample', 'suite_sha256'):
                if manifests['base'].get(key) != manifests['finetuned'].get(key):
                    raise ValueError(f'{split}: 两种模型的生成条件不同：{key}')
            results[split] = compare_split(rows, predictions['base'], predictions['finetuned'])
            provenance['splits'][split] = {
                'reference_path': str(data / f'{split}.jsonl'), 'reference_sha256': sha(data / f'{split}.jsonl'),
                'before_predictions': str(source / f'{split}-base.jsonl'),
                'after_predictions': str(source / f'{split}-finetuned.jsonl'),
                'before_manifest': manifests['base'], 'after_manifest': manifests['finetuned'],
            }
            # 保存新的逐题评分，原始预测与原有严格评分报告均不改写。
            for mode, key in (('base', 'before'), ('finetuned', 'after')):
                write_json(output / f'{split}-{mode}-scores.json', {
                    'strict': results[split]['summary'][f'strict_{key}'],
                    'diagnostic': results[split]['summary'][f'diagnostic_{key}'],
                    'results': [{'id': r['id'], 'strict': r[f'strict_{key}'],
                                 'diagnostic': r[f'diagnostic_{key}']} for r in results[split]['rows']]})
        stage = 'exporting_excel'
        write_json(output / 'comparison.json', {'provenance': provenance, 'results': results})
        write_json(status_path, {'status': 'running', 'stage': stage})
        exporter(output, node, modules)
        if not (output / '模型效果对比.xlsx').is_file():
            raise ValueError('Excel 未成功导出')
        write_json(status_path, {'status': 'completed', 'stage': 'completed',
                                'excel': '模型效果对比.xlsx', 'excel_sha256': sha(output / '模型效果对比.xlsx')})
        return results
    except BaseException as exc:
        write_json(status_path, {'status': 'failed', 'stage': stage, 'error': str(exc)})
        raise
