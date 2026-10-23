"""验证直接编写的 100 题答案；不调用任何模型服务。"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from pyforge.evaluation import docker_environment, run_isolated
from verify_teacher_data import CHECKS, HELPER


EXTRA = {
1: "r=two_sum([3,3],6); assert r[0]!=r[1] and sorted(r)==[0,1]",
11: "a=[[2,3],[1,2]]; assert merge_intervals(a)==[[1,3]] and a==[[2,3],[1,2]]",
19: "from itertools import product; texts=[''.join(p) for n in range(5) for p in product('ab',repeat=n)]; assert all(substring_index(t,p)==t.find(p) for t in texts for p in texts[:15])",
22: "assert [n for n in range(30) if is_prime(n)]==[2,3,5,7,11,13,17,19,23,29]",
40: "from itertools import product; assert all(inversion_count(a)==sum(a[i]>a[j] for i in range(len(a)) for j in range(i+1,len(a))) for a in product(range(3),repeat=5))",
46: "assert word_counts('KİA')=={'a':1}",
61: "calls=[]; f=memoize(lambda x:(calls.append(x),x*2)[1]); assert f(2)==f(2)==4 and calls==[2]",
80: "from decimal import Decimal; assert add_money('100000000000000000000000000000.1','0.2')==Decimal('100000000000000000000000000000.3')",
}

REFERENCES = {87:41,88:42,89:11,90:13,91:60,92:50,93:44}
MUTANTS = {
86: 'def clamp(value, low, high): return value',
87: 'def parse_ints(text): return []',
88: 'def chunked(items, size): return [items]',
89: 'def merge_intervals(intervals): return sorted(intervals)',
90: 'def first_unique(text): return 0',
91: 'def retry_call(func, attempts): return func()',
92: 'def write_json(path, value): pass',
93: 'def dedupe_records(records, key): return records',
}


def pytest_worker(reference, tests, should_pass):
    return '''import sys, types
from pathlib import Path
import pytest
_module = types.ModuleType('solution')
exec(%r, _module.__dict__)
sys.modules['solution'] = _module
class Solution:
    def verify(self):
        path = Path('/tmp/test_solution.py')
        path.write_text(%r, encoding='utf-8')
        exit_code = pytest.main(['-q', '--capture=sys', '-p', 'no:cacheprovider', str(path)])
        return (int(exit_code) == 0) is %r
''' % (reference, tests, should_pass)


def verify(source, image='pyforge-eval:py312-pytest835'):
    source=Path(source)
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    if len(rows)!=100 or any(r.get('source')!='codex_authored_v1' for r in rows):
        raise ValueError('需要完整的 100 题 Codex 编写数据')
    image_id=docker_environment(image)
    codes={int(r['id'].split('-')[-1]):re.findall(r'```python\n(.*?)```',r['messages'][-1]['content'],re.S)[0] for r in rows}
    manifest={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'image_id':image_id,
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_checks_sha256':hashlib.sha256(Path(__file__).with_name('verify_teacher_data.py').read_bytes()).hexdigest()}
    root=source.parent
    path=root/'verification_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('检查配置变化，请使用新数据版本')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    journal=root/'verification.jsonl'
    done={r['id']:r for r in map(json.loads,journal.read_text(encoding='utf-8').splitlines())} if journal.exists() else {}
    updated=[]
    for row in rows:
        number=int(row['id'].split('-')[-1])
        if row['id'] in done:
            result=done[row['id']]
        else:
            if number<=85:
                checks=CHECKS[number]+('; '+EXTRA[number] if number in EXTRA else '')
                code=codes[number]+'\n'+HELPER.replace('{checks}',checks)
                check_kind='independent_function_cases'
            elif number<=93:
                reference=codes[REFERENCES[number]] if number in REFERENCES else 'def clamp(value, low, high):\n if low>high: raise ValueError()\n return max(low,min(value,high))'
                code=pytest_worker(reference,codes[number],True)
                check_kind='pytest_correct_implementation_and_mutant'
            else:
                code=codes[number]+'\nclass Solution:\n    def verify(self): return True\n'
                check_kind='explanation_executable_assertions'
            problem={'method':'verify','comparator':'exact','tests':[{'args':[],'expected':True}]}
            result={'id':row['id'],'kind':check_kind,**run_isolated(code,problem,image_id,timeout=15)}
            if 86<=number<=93 and result['status']=='passed':
                mutation=run_isolated(pytest_worker(MUTANTS[number],codes[number],False),problem,image_id,timeout=15)
                result['mutant_detected']=mutation['status']=='passed'
                if not result['mutant_detected']:
                    result['status']='mutation_not_detected'
            with journal.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n')
        row['verification']={'syntax':True,'execution':result['status']=='passed','check_kind':result['kind'],
            'human_review':False,'mutant_detected':result.get('mutant_detected')}
        row['status']='execution_checked' if result['status']=='passed' else 'pending_review'
        updated.append(row)
        print(row['id'],result['status'],flush=True)
    for split in ('train','validation','test','pending_review'):
        selected=[r for r in updated if (not r['verification']['execution'] if split=='pending_review' else r['split']==split and r['verification']['execution'])]
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in selected),encoding='utf-8',newline='\n')
    summary={'total':len(rows),'verified':sum(r['verification']['execution'] for r in updated),
        'splits':{s:sum(r['split']==s and r['verification']['execution'] for r in updated) for s in ('train','validation','test')},
        'external_model_api_calls':0,'author':'Codex','files_sha256':{s:hashlib.sha256((root/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','test','pending_review')},
        'limitations':'通过自有功能用例、示例断言与指定错误实现检查，不是形式化证明，也未进行人工审核'}
    (root/'dataset_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/codex_build/candidates.jsonl')
    args=parser.parse_args()
    print(json.dumps(verify(args.source),ensure_ascii=False,indent=2))
