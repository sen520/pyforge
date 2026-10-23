"""冻结留出集：离线生成基础/适配器答案，再用 Docker 执行现有用例。"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

from pyforge.data import read_records
from pyforge.evaluation import docker_environment, extract_code, run_isolated
from pyforge.training import configure_hub_runtime, load_base, write_json
from verify_codex_data import MUTANTS, pytest_worker
from verify_teacher_data import CHECKS, HELPER


SUPPORTED = {'seed-016', 'seed-050', 'seed-053', 'seed-058', 'seed-092'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_worker(reference, code, should_pass):
    # 题面没有规定被测文件名，接受 solution.py 和 write_json.py 两种导入。
    return pytest_worker(reference, code, should_pass).replace(
        "sys.modules['solution'] = _module",
        "sys.modules['solution'] = _module\nsys.modules['write_json'] = _module")


def load_test(run, source):
    manifest = json.loads((Path(run) / 'run_manifest.json').read_text(encoding='utf-8'))
    if digest(source) != manifest['data_sha256']['test']:
        raise ValueError('留出集与训练清单的哈希不一致')
    rows = read_records(source)
    if {r['id'] for r in rows} != SUPPORTED:
        raise ValueError('当前执行检查仅支持冻结的 5 道留出题')
    return manifest, rows


def generate(run, source, output, max_tokens):
    manifest, rows = load_test(run, source)
    if max_tokens <= 0:
        raise ValueError('max-new-tokens 必须为正整数')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configure_hub_runtime(True)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed

    model, _, _ = load_base(manifest['config']['model'], manifest['model_revision'], True)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(run) / 'adapter'), local_files_only=True)
    model = PeftModel.from_pretrained(model, str(Path(run) / 'adapter'))
    model.eval()
    model.config.use_cache = True
    metadata = {'run': str(Path(run).resolve()), 'model_revision': manifest['model_revision'],
        'test_sha256': digest(source), 'adapter_sha256': digest(Path(run) / 'adapter/adapter_model.safetensors'),
        'run_manifest_sha256': digest(Path(run) / 'run_manifest.json'),
        'script_sha256': digest(__file__), 'max_new_tokens': max_tokens, 'do_sample': False,
        'versions': {name: __import__('importlib.metadata', fromlist=['version']).version(name)
                     for name in manifest['versions']}}
    write_json(output / 'generation_manifest.json', metadata)
    for mode in ('base', 'finetuned'):
        with (output / (mode + '_predictions.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                # 不把参考回答、测试代码或检查记录传给模型。
                messages = row['messages'][:-1]
                if any(m['role'] == 'assistant' for m in messages):
                    raise ValueError('此留出流程只支持单轮题目')
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_tensors='pt').to(model.device)
                if inputs.shape[1] > manifest['config']['max_length']:
                    raise ValueError('题目超过训练输入长度')
                if inputs.shape[1] + max_tokens > model.config.max_position_embeddings:
                    raise ValueError('题目与生成长度超过模型上下文')
                set_seed(manifest['config']['seed'])
                began = time.monotonic()
                with torch.inference_mode(), (model.disable_adapter() if mode == 'base' else nullcontext()):
                    tokens = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                        temperature=None, top_p=None, top_k=None, max_new_tokens=max_tokens,
                        pad_token_id=tokenizer.pad_token_id)[0, inputs.shape[1]:]
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                result = {'id': row['id'], 'response': tokenizer.decode(tokens, skip_special_tokens=True),
                    'generated_tokens': len(tokens), 'truncated': int(tokens[-1]) not in eos,
                    'elapsed_seconds': time.monotonic() - began}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                print(mode, row['id'], 'tokens=', len(tokens), 'truncated=', result['truncated'], flush=True)


def score(run, source, output, image):
    _, rows = load_test(run, source)
    output = Path(output)
    generation = json.loads((output / 'generation_manifest.json').read_text(encoding='utf-8'))
    if generation['test_sha256'] != digest(source) or generation['adapter_sha256'] != digest(Path(run) / 'adapter/adapter_model.safetensors'):
        raise ValueError('生成后数据或适配器发生变化')
    if any((output / (mode + '_report.json')).exists() for mode in ('base', 'finetuned')):
        raise ValueError('评估报告已存在，拒绝覆盖')
    image_id = docker_environment(image)
    problem = {'method': 'verify', 'comparator': 'exact', 'tests': [{'args': [], 'expected': True}]}
    reference = extract_code(next(r for r in rows if r['id'] == 'seed-050')['messages'][-1]['content'])
    # 先验证检查器与正确参考实现能运行，避免把环境失败误记为模型失败。
    for row in rows:
        code = extract_code(row['messages'][-1]['content'])
        number = int(row['id'].split('-')[-1])
        worker = test_worker(reference, code, True) if number == 92 else code + '\n' + HELPER.replace('{checks}', CHECKS[number])
        if run_isolated(worker, problem, image_id, timeout=15)['status'] != 'passed':
            raise RuntimeError('参考实现自检失败：' + row['id'])
    for mode in ('base', 'finetuned'):
        path = output / (mode + '_predictions.jsonl')
        predictions = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(predictions) != len(rows) or {r['id'] for r in predictions} != SUPPORTED:
            raise ValueError('预测 ID 缺失或重复')
        results = []
        for prediction in predictions:
            number = int(prediction['id'].split('-')[-1])
            result = {'id': prediction['id'], 'truncated': prediction['truncated']}
            try:
                code = extract_code(prediction['response'])
            except (ValueError, SyntaxError) as exc:
                result.update(status='extraction_or_syntax_error', error=str(exc))
            else:
                worker = test_worker(reference, code, True) if number == 92 else code + '\n' + HELPER.replace('{checks}', CHECKS[number])
                result.update(run_isolated(worker, problem, image_id, timeout=15))
                if number == 92 and result['status'] == 'passed':
                    mutation = run_isolated(test_worker(MUTANTS[92], code, False), problem, image_id, timeout=15)
                    result['mutant_detected'] = mutation['status'] == 'passed'
                    if not result['mutant_detected']:
                        result['status'] = 'mutation_not_detected'
            results.append(result)
            print(mode, result, flush=True)
        passed = sum(r['status'] == 'passed' for r in results)
        report = {'mode': mode, 'total': len(rows), 'passed': passed, 'pass_at_1': passed / len(rows),
            'results': results, 'test_sha256': digest(source), 'predictions_sha256': digest(path),
            'generation': generation, 'image_id': image_id, 'timeout_seconds': 15,
            'scorer_sha256': digest(__file__),
            'checks_sha256': digest(Path(__file__).with_name('verify_teacher_data.py')),
            'pytest_worker_sha256': digest(Path(__file__).with_name('verify_codex_data.py')),
            'scope': '5 题有限用例；复用已有数据验证器，不代表官方或全面正确率'}
        write_json(output / (mode + '_report.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['generate', 'score'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', default='datasets/test.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--image', default='pyforge-eval:py312-pytest835')
    args = parser.parse_args()
    if args.phase == 'generate':
        generate(args.run, args.data, args.output, args.max_new_tokens)
    else:
        score(args.run, args.data, args.output, args.image)
