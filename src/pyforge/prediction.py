"""为可执行算法题库生成原始预测；模型只读取 prompt，不读取 tests 或参考答案。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from pyforge.training import configure_hub_runtime, load_base, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(run, suite, output, data='datasets', max_tokens=1024, base=False, limit=None):
    if max_tokens <= 0 or (limit is not None and limit <= 0):
        raise ValueError('生成长度和 limit 必须为正整数')
    run, suite, output = Path(run), Path(suite), Path(output)
    sidecar = output.with_suffix('.manifest.json')
    if output.exists() or sidecar.exists():
        raise ValueError('预测文件或清单已存在，请使用新文件名')
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    train = Path(data) / 'train.jsonl'
    if sha(train) != manifest['data_sha256']['train']:
        raise ValueError('--data 训练数据哈希与模型清单不一致，无法检查评估题重叠')
    training_ids = {json.loads(line)['id'] for line in train.read_text(encoding='utf-8').splitlines()}
    problems = [json.loads(line) for line in suite.read_text(encoding='utf-8').splitlines()]
    ids = [p['id'] for p in problems]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(p.get('prompt'), str) or not p['prompt'].strip() for p in problems):
        raise ValueError('题库必须有唯一 ID 与非空 prompt')
    if training_ids & set(ids):
        raise ValueError('评估题 ID 与训练集重叠')
    if limit is not None:
        problems = problems[:limit]
    configure_hub_runtime(True)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed

    set_seed(manifest['config']['seed'])
    model, _, _ = load_base(manifest['config']['model'], manifest['model_revision'], True)
    tokenizer = AutoTokenizer.from_pretrained(str(run / 'adapter'), local_files_only=True, trust_remote_code=False)
    if not base:
        model = PeftModel.from_pretrained(model, str(run / 'adapter'))
    model.eval()
    model.config.use_cache = True
    # 在写文件前检查全部题面长度，避免中途才发现输入不适用。
    encoded = []
    for problem in problems:
        messages = ([{'role': 'system', 'content': manifest['system_prompt']}] if manifest.get('system_prompt') else [])
        messages.append({'role': 'user', 'content': problem['prompt']})
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        if tokens.shape[1] > manifest['config']['max_length'] or tokens.shape[1] + max_tokens > model.config.max_position_embeddings:
            raise ValueError('输入或生成长度超出限制：' + problem['id'])
        encoded.append(tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'run': str(run.resolve()), 'mode': 'base' if base else 'finetuned',
        'suite_sha256': sha(suite), 'run_manifest_sha256': sha(run / 'run_manifest.json'),
        'adapter_sha256': None if base else sha(run / 'adapter/adapter_model.safetensors'),
        'model_revision': manifest['model_revision'], 'script_sha256': sha(__file__),
        'max_new_tokens': max_tokens, 'do_sample': False, 'count': len(problems), 'status': 'generating'}
    write_json(sidecar, provenance)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        for problem, tokens in zip(problems, encoded):
            inputs = tokens.to(model.device)
            began = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    temperature=None, top_p=None, top_k=None, max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.pad_token_id)[0, inputs.shape[1]:]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            row = {'id': problem['id'], 'response': tokenizer.decode(generated, skip_special_tokens=True),
                'generated_tokens': len(generated), 'truncated': int(generated[-1]) not in eos,
                'elapsed_seconds': time.monotonic() - began}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(row['id'], 'tokens=', row['generated_tokens'], 'truncated=', row['truncated'], flush=True)
    write_json(sidecar, {**provenance, 'status': 'completed', 'predictions_sha256': sha(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--suite', default='datasets/benchmarks/doocs-leetcode/executable_v1.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default='datasets')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--base', action='store_true', help='不加载适配器，生成同一基础模型的基线预测')
    parser.add_argument('--limit', type=int, help='只生成前 N 题，用于检查流程；完整评估不要设置')
    args = parser.parse_args()
    predict(args.run, args.suite, args.output, args.data, args.max_new_tokens, args.base, args.limit)


if __name__ == '__main__':
    main()
