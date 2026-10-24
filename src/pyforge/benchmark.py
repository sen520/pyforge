"""一次运行两边预测、同条件执行和逐题比较；预测在独立进程中释放 GPU。"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from .comparison import compare
from .evaluation import docker_environment, evaluate
from .paths import PROJECT_ROOT
from .training import write_json


DEFAULT_SUITE = PROJECT_ROOT / 'datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl'


def generate(run, suite, output, data, max_tokens, base):
    command = [sys.executable, '-m', 'pyforge.prediction', '--run', str(run), '--suite', str(suite),
        '--output', str(output), '--data', str(data), '--max-new-tokens', str(max_tokens)]
    if base:
        command.append('--base')
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(('基础' if base else '微调') + '模型生成失败，退出码：' + str(result.returncode))


def benchmark(run, output, suite=DEFAULT_SUITE, data='datasets/leetcode', max_tokens=1024,
              image='python:3.12-slim', timeout=10):
    if type(max_tokens) is not int or max_tokens <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('生成长度和执行超时必须为正数')
    run, output, suite, data = (Path(p).resolve() for p in (run, output, suite, data))
    if output.exists():
        raise ValueError('评估输出目录已存在，请使用新目录')
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer_config.json'):
        if not (run / 'adapter' / name).is_file():
            raise ValueError('缺少训练适配器文件：' + name)
    train_file = data / 'train.jsonl'
    if hashlib.sha256(train_file.read_bytes()).hexdigest() != manifest['data_sha256']['train']:
        raise ValueError('训练数据哈希与模型清单不一致，请使用原数据')
    training_ids = {json.loads(line)['id'] for line in train_file.read_text(encoding='utf-8').splitlines()}
    problems = [json.loads(line) for line in suite.read_text(encoding='utf-8').splitlines()]
    ids = [p['id'] for p in problems]
    if not ids or len(set(ids)) != len(ids) or any(not p.get('execution_ready') or not p.get('tests')
        or not isinstance(p.get('prompt'), str) or not p['prompt'].strip() for p in problems):
        raise ValueError('题库必须非空、ID 唯一，且包含题面和可执行用例')
    if training_ids & set(ids):
        raise ValueError('评估题与训练题 ID 重叠')
    # 先确认 Docker 和本地镜像，避免耗时生成后才发现无法执行；固定镜像 ID。
    image_id = docker_environment(image)
    output.mkdir(parents=True)
    status_path = output / 'pipeline_status.json'
    stage = 'starting'
    try:
        steps = [('generating_base', '1/5 生成基础模型答案'),
                 ('generating_finetuned', '2/5 生成微调模型答案'),
                 ('evaluating_base', '3/5 执行基础模型测试'),
                 ('evaluating_finetuned', '4/5 执行微调模型测试'),
                 ('comparing', '5/5 生成逐题对比表')]
        for stage, label in steps:
            print(label, flush=True)
            write_json(status_path, {'status': 'running', 'stage': stage, 'problem_count': len(ids)})
            if stage.startswith('generating_'):
                base = stage == 'generating_base'
                name = 'base' if base else 'finetuned'
                generate(run, suite, output / (name + '_predictions.jsonl'), data, max_tokens, base)
            elif stage.startswith('evaluating_'):
                name = 'base' if stage == 'evaluating_base' else 'finetuned'
                evaluate(suite, output / (name + '_predictions.jsonl'), output / (name + '_report.json'), image_id, timeout)
            else:
                summary = compare(suite, output / 'base_predictions.jsonl', output / 'base_report.json',
                    output / 'finetuned_predictions.jsonl', output / 'finetuned_report.json', output / 'comparison')
                print((output / 'comparison/逐题对比.md').read_text(encoding='utf-8'), flush=True)
        result = {'status': 'completed', 'summary': summary,
            'table': str(output / 'comparison/逐题对比.md'),
            'answers_table': str(output / 'comparison/逐题答案对比.html'),
            'data': str(output / 'comparison/comparison.json')}
        write_json(status_path, result)
        return result
    except BaseException as exc:
        write_json(status_path, {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
