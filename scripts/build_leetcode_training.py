"""将固定 doocs 快照的已有 Python 题解转为可追溯 SFT 数据，不调用模型 API。"""

import argparse
import ast
import builtins
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import symtable
import sys

from pyforge.data import read_records, split_records
from pyforge.encoding import encode_record
from pyforge.leetcode import extract_problem, replace_interface
from pyforge.paths import MODEL_CACHE_DIR
from pyforge.training import configure_hub_runtime, resolve_snapshot


IMPORTS = {}
for module, names in {
    'typing': 'List Optional Tuple Dict Set Deque DefaultDict Callable Any Iterator Iterable',
    'collections': 'Counter defaultdict deque',
    'functools': 'cache lru_cache reduce cmp_to_key partial',
    'itertools': 'accumulate pairwise combinations permutations product chain groupby count cycle repeat zip_longest',
    'math': 'inf gcd lcm sqrt isqrt ceil floor comb factorial perm log log2 log10 prod pow',
    'heapq': 'heappush heappop heapify heappushpop heapreplace nlargest nsmallest merge',
    'bisect': 'bisect_left bisect_right bisect insort insort_left insort_right',
    'operator': 'xor or_ and_ itemgetter',
    'string': 'ascii_lowercase ascii_uppercase digits',
    'random': 'randint randrange choice shuffle',
}.items():
    for name in names.split():
        IMPORTS[name] = f'from {module} import {name}'
for module in ('math', 'heapq', 'bisect', 'collections', 'functools', 'itertools', 'random', 'string', 'operator', 're'):
    IMPORTS[module] = 'import ' + module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(code):
    tree = ast.parse(code)
    if any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)) for n in tree.body):
        raise ValueError('顶层包含执行语句')
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            modules = [a.name.split('.')[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            modules = [(n.module or '').split('.')[0]]
        else:
            continue
        if any(m not in sys.stdlib_module_names for m in modules):
            raise ValueError('非标准库依赖')
    table = symtable.symtable(code, '<answer>', 'exec')
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    def referenced(t):
        return {s.get_name() for s in t.get_symbols() if s.is_referenced() and s.is_global()} | set().union(*(referenced(c) for c in t.get_children()))
    missing = referenced(table) - defined - set(dir(builtins)) - {'__name__'}
    unresolved = missing - IMPORTS.keys()
    if unresolved:
        raise ValueError('未适配全局依赖：' + ','.join(sorted(unresolved)))
    prelude = '\n'.join(sorted({IMPORTS[name] for name in missing}))
    result = (prelude + '\n\n' if prelude else '') + code.strip() + '\n'
    compile(result, '<answer>', 'exec')
    return result


def answer_from_readme(raw, interface):
    expected = ast.parse(interface).body[0]
    names = {m.name for m in expected.body if isinstance(m, ast.FunctionDef)}
    errors = []
    for code in re.findall(r'```python\s*\n(.*?)```', raw, re.S):
        try:
            result = normalize(code)
            matching = next((n for n in ast.parse(result).body if isinstance(n, ast.ClassDef) and n.name == expected.name), None)
            if matching is None or not names <= {n.name for n in matching.body if isinstance(n, ast.FunctionDef)}:
                raise ValueError('接口不匹配')
            return result
        except (ValueError, SyntaxError) as exc:
            errors.append(str(exc))
    raise ValueError('; '.join(sorted(set(errors))) or '没有 Python 题解')


def build(source, output, legacy, max_length=2048):
    configure_hub_runtime(True)
    from transformers import AutoTokenizer
    source, output, legacy = Path(source), Path(output), Path(legacy)
    if output.exists():
        raise ValueError('输出目录已存在，拒绝覆盖')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    benchmark = Path('datasets/leetcode/benchmarks/doocs-leetcode')
    original = json.loads((benchmark / 'manifest.json').read_text(encoding='utf-8'))
    if revision != original['source_revision']:
        raise ValueError('快照版本与冻结题库不一致')
    reserved = {r['id'] for r in map(json.loads, (benchmark / 'executable_v1.jsonl').read_text(encoding='utf-8').splitlines())}
    candidates = [json.loads(line) for line in (benchmark / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    snapshot, tokenizer_revision = resolve_snapshot('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'main', True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    rows, skipped, code_groups = [], [], {}
    for question in candidates:
        if question['id'] in reserved:
            skipped.append({'id': question['id'], 'reason': 'reserved_executable_benchmark'})
            continue
        if question['contains_images']:
            skipped.append({'id': question['id'], 'reason': 'image_dependency'})
            continue
        try:
            path = source / question['source_path']
            raw = path.read_text(encoding='utf-8')
            if hashlib.sha256(raw.encode()).hexdigest() != question['source_sha256']:
                raise ValueError('题面来源哈希变化')
            interface = extract_problem(raw)['interface']
            prompt = replace_interface(question['prompt'], question['interface'], interface)
            code = answer_from_readme(raw, interface)
            canonical = ast.dump(ast.parse(code), include_attributes=False)
            code_hash = hashlib.sha256(canonical.encode()).hexdigest()
            group = code_groups.setdefault(code_hash, question['group_id'])
            row = {'id': question['id'], 'group_id': group, 'task_type': 'algorithm', 'language': 'zh',
                'source': 'doocs_reference_sft_v1', 'author': 'doocs/leetcode contributors',
                'authorship': 'upstream_reference_not_codex_authored', 'license': question['license'],
                'attribution': question['attribution'], 'source_revision': revision,
                'source_path': question['source_path'], 'source_url': question['source_url'],
                'source_sha256': question['source_sha256'], 'problem_id': question['problem_id'],
                'difficulty': question['difficulty'], 'tags': question['tags'],
                'changes': '提取题面和已有 Python 题解；补全可识别标准库导入；不复制推理过程；按完整对话长度筛选',
                'status': 'syntax_checked_not_execution_verified',
                'verification': {'syntax': True, 'dependency_static': True, 'execution': False, 'human_review': False},
                'messages': [
                    {'role': 'system', 'content': '你是 Python 编程助手。遵守接口与约束，用简短说明和完整代码回答。'},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': '```python\n' + code + '```'}]}
            value = encode_record(row, tokenizer, max_length)
            if value is None:
                raise ValueError('完整问答超过 max_length')
            row['token_length'] = len(value['input_ids'])
            rows.append(row)
        except (ValueError, SyntaxError) as exc:
            skipped.append({'id': question['id'], 'reason': str(exc)})
    splits = split_records(rows)
    # 原来的已执行种子训练/验证划分不变；已查看过的 5 题留出集仅存档。
    for split in ('train', 'validation'):
        splits[split] = read_records(legacy / (split + '.jsonl')) + splits[split]
    for split, values in splits.items():
        for row in values:
            row['split'] = split
    # 不同题号若实现 AST 相同则共用题族；另外仍保留种子近似题排除规则。
    output.mkdir(parents=True)
    (output / 'LICENSE').write_bytes((source / 'LICENSE').read_bytes())
    files = {}
    for name, values in splits.items():
        path = output / (name + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8', newline='\n')
        read_records(path)
        files[name] = {'count': len(values), 'sha256': sha(path)}
    manifest = {'source_revision': revision, 'builder_sha256': sha(__file__), 'max_length': max_length,
        'tokenizer': 'Qwen/Qwen2.5-Coder-0.5B-Instruct', 'tokenizer_revision': tokenizer_revision, 'upstream_answers_added': len(rows),
        'retained_legacy_train': 90, 'retained_legacy_validation': 5, 'legacy_test_archived': 5,
        'files': files, 'difficulty': dict(Counter(r['difficulty'] for r in rows)),
        'reserved_benchmark_ids': sorted(reserved), 'skipped': skipped,
        'source_questions_sha256': sha(benchmark / 'questions.jsonl'),
        'external_model_api_calls': 0, 'execution_verified_upstream_answers': 0,
        'limitations': ['上游答案仅静态检查，不能等同执行正确', '相同 AST 和已知种子近似题已分组/排除，未完整语义去污染',
            '公开题可能出现在预训练中', '旧完整题面库改作来源目录，不再整体独立评估；仅保留指定评估题和新划分']}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/sources/doocs-leetcode')
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy', default='datasets/leetcode/archive/codex_v1')
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    result = build(args.source, args.output, args.legacy, args.max_length)
    print(json.dumps({k: result[k] for k in ('upstream_answers_added', 'files', 'difficulty')}, ensure_ascii=False, indent=2))
