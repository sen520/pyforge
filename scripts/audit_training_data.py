"""固定分层抽样并在 Docker 中执行训练答案；不修改正式三个数据集。"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from pyforge.evaluation import docker_environment, extract_code, run_isolated
from audit_boundary_cases import CASES, COMPARATORS

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sample_rows(rows, count=200):
    groups = defaultdict(list)
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('审计禁止使用测试集')
        groups[(row['split'], row['difficulty'])].append(row)
    total = len(rows)
    if not 0 < count <= total:
        raise ValueError('抽样数量不合法')
    quota = {key: len(group) * count // total for key, group in groups.items()}
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * count % total), key))
    for key in order[:count - sum(quota.values())]:
        quota[key] += 1
    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda r: digest(('audit-v1:42:' + r['id']).encode()))[:quota[key]])
    return selected


def literal(text):
    tree = ast.parse(text.strip(), mode='eval')
    class Constants(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in ('true', 'false', 'null'):
                raise ValueError('不是 JSON/Python 字面量')
            return ast.copy_location(ast.Constant({'true': True, 'false': False, 'null': None}[node.id]), node)
    return ast.literal_eval(Constants().visit(tree))


def examples(description, parameters):
    cases = []
    for match in re.finditer(r'输入[：:]\s*(.*?)\s*输出[：:]\s*(.*?)(?=\n\s*(?:解释|说明|示例|提示|约束)|\Z)', description, re.S):
        try:
            call = ast.parse('f(' + match[1].strip() + ')', mode='eval').body
            if not isinstance(call, ast.Call) or call.args or any(k.arg is None for k in call.keywords):
                continue
            given = {k.arg: literal(ast.unparse(k.value)) for k in call.keywords}
            if set(given) != set(parameters):
                continue
            # 输出只接受完整字面量；不从解释文字或数字前缀猜结果。
            lines = match[2].strip().splitlines()
            expected = None
            parsed = False
            for end in range(1, len(lines) + 1):
                try:
                    expected = literal('\n'.join(lines[:end]))
                    parsed = True
                    break
                except (SyntaxError, ValueError, TypeError):
                    pass
            if parsed:
                cases.append({'args': [given[p] for p in parameters], 'expected': expected,
                              'origin': 'upstream_problem_example'})
        except (SyntaxError, ValueError, TypeError):
            continue
    return cases


# 该代码仅作为字符串交给 Docker，宿主不执行候选答案。
CHECKER = '''
import math
from collections import Counter
from copy import deepcopy
def match(a, b, mode, args):
    if mode == 'unordered':
        return isinstance(a, (list, tuple)) and Counter(map(repr, a)) == Counter(map(repr, b))
    if mode == 'peak_index':
        return type(a) is int and 0 <= a < len(args[0]) and (a == 0 or args[0][a] > args[0][a-1]) and (a == len(args[0])-1 or args[0][a] > args[0][a+1])
    if mode == 'no_zero_pair':
        return isinstance(a, (list, tuple)) and len(a) == 2 and all(type(x) is int and x > 0 and '0' not in str(x) for x in a) and sum(a) == args[0]
    if mode == 'distance_grid':
        r, c, x, y = args
        if not isinstance(a, (list, tuple)) or len(a) != r*c or any(not isinstance(p, (list, tuple)) or len(p) != 2 or any(type(v) is not int for v in p) for p in a): return False
        if set(map(tuple, a)) != {(i,j) for i in range(r) for j in range(c)}: return False
        d = [abs(i-x)+abs(j-y) for i,j in a]
        return d == sorted(d)
    if mode == 'smallest_pairs':
        u,v,k = args
        pool = Counter((x,y) for x in u for y in v)
        if not isinstance(a, (list, tuple)) or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in a): return False
        found = Counter(map(tuple, a))
        return len(a) == min(k,len(u)*len(v)) and not (found-pool) and sorted(sum(p) for p in a) == sorted(x+y for x in u for y in v)[:k]
    if mode == 'triplet_partition':
        nums,k = args
        if a == []:
            s=sorted(nums)
            return any(s[i+2]-s[i] > k for i in range(0,len(s),3))
        return isinstance(a, (list, tuple)) and all(isinstance(g, (list, tuple)) and len(g)==3 and max(g)-min(g)<=k for g in a) and Counter(x for g in a for x in g)==Counter(nums)
    if isinstance(b, bool): return type(a) is bool and a == b
    if isinstance(b, (int, float)):
        return type(a) in (int,float) and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6)
    if isinstance(b, list):
        return isinstance(a, (list,tuple)) and len(a)==len(b) and all(match(x,y,'exact',[]) for x,y in zip(a,b))
    return type(a) is type(b) and a==b
'''


def wrapper(code, method):
    return CHECKER + '\nns={"__name__":"__candidate__"}\nexec(' + repr(code) + ',ns)\n' + '''
class Solution:
    def verify(self, args, expected, mode):
        actual = getattr(ns['Solution'](), METHOD)(*deepcopy(args))
        return bool(match(actual, expected, mode, args))
'''.replace('METHOD', repr(method))


def prepare(row, question):
    result = {k: row[k] for k in ('id', 'split', 'difficulty', 'problem_id', 'source_url', 'source_revision')}
    result['answer_sha256'] = digest(row['messages'][-1]['content'].encode())
    result['title'] = question['title']
    try:
        tree = ast.parse(question['interface'])
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solution')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]
        if len(methods) != 1:
            raise ValueError('需要单个公开方法')
        method = methods[0]
        result['method'] = method.name
        parameters = [arg.arg for arg in method.args.args if arg.arg != 'self']
        cases = examples(question['description'], parameters)
        cases += [{'args': a, 'expected': b, 'origin': 'codex_independent_contract_cases'} for a,b in CASES.get(row['problem_id'], [])]
        # 原地修改/链表/树/交互题需另写适配器，绝不当作答案错误。
        annotation = question['interface']
        if any(name in annotation for name in ('ListNode', 'TreeNode', 'Node', 'Iterator', 'MountainArray', 'NestedInteger')):
            raise ValueError('外部对象接口尚未适配')
        if not cases or any(c['expected'] is None for c in cases) and row['problem_id'] not in COMPARATORS:
            raise ValueError('没有可直接检查的返回值示例')
        result['tests'] = cases
        result['comparator'] = COMPARATORS.get(row['problem_id'], 'exact')
        result['status'] = 'ready'
    except (ValueError, SyntaxError, StopIteration) as exc:
        result['status'] = 'unsupported_adapter'
        result['reason'] = str(exc) or '不支持此类接口'
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def audit(output, execute=False):
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for split in ('train', 'validation'):
        path = ROOT / 'datasets' / (split + '.jsonl')
        hashes[split] = digest(path.read_bytes())
        rows.extend(r for r in map(json.loads, path.read_text(encoding='utf8').splitlines()) if r['source'] == 'doocs_reference_sft_v1')
    selected = sample_rows(rows)
    questions = {q['id']: q for q in map(json.loads, (ROOT / 'datasets/benchmarks/doocs-leetcode/questions.jsonl').read_text(encoding='utf8').splitlines())}
    items = [prepare(row, questions[row['id']]) for row in selected]
    manifest = {'version': 1, 'seed': 42, 'sampling': 'proportional_split_difficulty_largest_remainder_sha256_rank',
                'population': len(rows), 'sample_size': len(items), 'dataset_sha256': hashes,
                'author': 'Codex', 'human_review': False, 'items': items}
    frozen = output / 'sample.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf8')) != manifest:
        raise ValueError('样本/检查器已变化，请使用新的输出目录，禁止覆盖旧审计')
    write_json(frozen, manifest)
    print('fixed_sample=', len(items), 'ready=', sum(i['status']=='ready' for i in items), flush=True)
    if not execute:
        return
    image = docker_environment('python:3.12-slim')
    identity = {'sample_sha256': digest(frozen.read_bytes()), 'checker_sha256': digest(CHECKER.encode()),
                'runner_sha256': digest(Path(__file__).read_bytes()), 'image_id': image, 'timeout_seconds': 15}
    identity_path = output / 'execution_manifest.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf8')) != identity:
        raise ValueError('执行条件发生变化，请使用新输出目录')
    write_json(identity_path, identity)
    journal = output / 'results.jsonl'
    done = {r['id']: r for r in map(json.loads, journal.read_text(encoding='utf8').splitlines())} if journal.exists() else {}
    answers = {r['id']: r for r in selected}
    for index, item in enumerate(items, 1):
        if item['id'] in done:
            continue
        result = {'id': item['id'], 'split': item['split'], 'difficulty': item['difficulty']}
        if item['status'] != 'ready':
            result.update(status=item['status'], reason=item['reason'])
        else:
            code = extract_code(answers[item['id']]['messages'][-1]['content'])
            problem = {'method': 'verify', 'comparator': 'exact', 'tests': [
                {'args': [c['args'], c['expected'], item['comparator']], 'expected': True} for c in item['tests']]}
            result.update(run_isolated(wrapper(code, item['method']), problem, image, 15, include_details=True))
        with journal.open('a', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        done[item['id']] = result
        print(index, item['id'], result['status'], flush=True)
    counts = dict(Counter(r['status'] for r in done.values()))
    summary = {**identity, 'total': len(items), 'counts': counts,
               'upstream_examples': sum(sum(c['origin']=='upstream_problem_example' for c in i.get('tests',[])) for i in items),
               'independent_cases': sum(sum(c['origin']=='codex_independent_contract_cases' for c in i.get('tests',[])) for i in items),
               'independent_case_problems': sum(any(c['origin']=='codex_independent_contract_cases' for c in i.get('tests',[])) for i in items),
               'scope': '固定抽样；本地示例/边界用例；不等于官方隐藏测试；未使用测试集；未修改训练答案；无人工作品审核'}
    write_json(output / 'summary.json', summary)
    lines = ['# 训练答案执行审计', '', '固定抽样 200 题，来自训练/验证集；检查器不支持的样本保留在分母中。', '',
             '| 题号 | 题目 | 集合 | 难度 | 状态 | 通过用例/用例数 |', '|---|---|---|---|---|---|']
    for item in items:
        r = done[item['id']]
        lines.append(f"| {item['problem_id']} | {item['title']} | {item['split']} | {item['difficulty']} | {r['status']} | {r.get('passed_cases','-')}/{r.get('total_cases','-')} |")
    (output / '逐题审计.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/data-audit-v1')
    parser.add_argument('--execute', action='store_true', help='使用 Docker 实际执行；默认只冻结样本')
    args = parser.parse_args()
    audit(args.output, args.execute)
