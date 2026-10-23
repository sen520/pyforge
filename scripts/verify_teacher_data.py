"""独立检查教师函数答案；只在 Docker 中执行，保留未验证样本。"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from pyforge.evaluation import docker_environment, run_isolated


# 自有测试，不使用教师生成的测试作为正确性标准。
CHECKS = {
1: "r=two_sum([2,7,11],9); assert len(r)==2 and r[0]!=r[1] and sum([2,7,11][i] for i in r)==9; assert two_sum([],0) is None; assert two_sum([1],2) is None",
2: "a=[3,1,3,-1]; assert unique_sorted(a)==[-1,1,3] and a==[3,1,3,-1]; assert unique_sorted([])==[]",
3: "assert merge_sorted([],[])==[]; assert merge_sorted([1,2,2],[0,2])==[0,1,2,2,2]",
4: "assert binary_search([1,2,2,2,3],2)==1; assert binary_search([],1)==-1; assert binary_search([1],2)==-1",
5: "assert lower_bound([],1)==0; assert lower_bound([1,2,2,4],2)==1; assert lower_bound([1,2],3)==2",
6: "assert rotate_right([],2)==[]; assert rotate_right([1,2,3],4)==[3,1,2]; assert rotate_right([1],0)==[1]",
7: "assert move_zeros([0,1,0,2])==[1,2,0,0]; assert move_zeros([])==[]; assert move_zeros([0,0])==[0,0]",
8: "assert max_subarray([-3,-1,-2])==-1; assert max_subarray([1,-2,3,4])==7; raises(ValueError,lambda:max_subarray([]))",
9: "assert longest_increasing_run([])==0; assert longest_increasing_run([1,2,2,3,4])==3; assert longest_increasing_run([3,2,1])==1",
10: "assert product_except_self([])==[]; assert product_except_self([7])==[1]; assert product_except_self([1,2,3])==[6,3,2]; assert product_except_self([0,2,0])==[0,0,0]",
11: "assert merge_intervals([])==[]; assert list(map(list,merge_intervals([[3,4],[1,3],[6,6]])))==[[1,4],[6,6]]; assert list(map(list,merge_intervals([[1,2],[3,4]])))==[[1,2],[3,4]]",
12: "assert intersection([1,1,2],[2,2,1])==[1,2]; assert intersection([],[1])==[]",
13: "assert first_unique('aabb')==-1; assert first_unique('aab中b')==3; assert first_unique('')==-1",
14: "assert is_palindrome('A man, a plan, a canal: Panama'); assert is_palindrome('!'); assert not is_palindrome('ab')",
15: "assert anagram_groups(['eat','tea','bat','ate'])==[['eat','tea','ate'],['bat']]; assert anagram_groups([])==[]",
16: "assert longest_unique_substring('abcabcbb')==3; assert longest_unique_substring('')==0; assert longest_unique_substring('中a中b')==3",
17: "assert valid_brackets(''); assert valid_brackets('([])'); assert not valid_brackets('([)]'); raises(ValueError,lambda:valid_brackets('a'))",
18: "assert run_length_encode('aaabb')==[['a',3],['b',2]]; assert run_length_encode('')==[]",
19: "assert substring_index('abc','')==0; assert substring_index('ababa','aba')==0; assert substring_index('a','b')==-1",
20: "assert common_prefix([])==''; assert common_prefix(['flower','flow','flight'])=='fl'; assert common_prefix(['','x'])==''",
21: "assert gcd(0,0)==0; assert gcd(-12,8)==4; assert gcd(0,-3)==3",
22: "assert not is_prime(-1); assert not is_prime(1); assert is_prime(2); assert not is_prime(49); assert is_prime(97)",
23: "assert sieve(0)==[]; assert sieve(2)==[2]; assert sieve(10)==[2,3,5,7]",
24: "assert fast_power(0,0)==1; assert fast_power(-2,3)==-8; assert fast_power(2,10)==1024",
25: "assert popcount(0)==0; assert popcount(15)==4; assert popcount(1024)==1",
26: "assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(10)==55",
27: "assert climb_stairs(0)==1; assert climb_stairs(1)==1; assert climb_stairs(5)==8",
28: "assert min_coins([2],3)==-1; assert min_coins([1,2,5],11)==3; assert min_coins([],0)==0",
29: "assert knapsack([2,3,4],[4,5,6],5)==9; assert knapsack([],[],0)==0; assert knapsack([2],[5],1)==0",
30: "assert longest_common_subsequence('abcde','ace')==3; assert longest_common_subsequence('','x')==0; assert longest_common_subsequence('a','b')==0",
31: "assert edit_distance('horse','ros')==3; assert edit_distance('','abc')==3; assert edit_distance('same','same')==0",
32: "assert grid_paths(0,3)==0; assert grid_paths(1,1)==1; assert grid_paths(3,7)==28",
33: "g={'a':['b','c'],'b':['d'],'c':['d']}; r=shortest_path(g,'a','d'); assert r in (['a','b','d'],['a','c','d']); assert shortest_path(g,'x','y') is None; assert shortest_path({},'x','x')==['x']",
34: "assert connected_components(0,[])==0; assert connected_components(4,[(0,1),(0,1),(2,2)])==3",
35: "r=topological_order(3,[(0,1),(1,2)]); assert r==[0,1,2]; assert topological_order(2,[(0,1),(1,0)]) is None; assert topological_order(0,[])==[]",
36: "a=[[1,1],[1,0]]; assert flood_fill(a,0,0,2)==[[2,2],[2,0]]; assert a==[[1,1],[1,0]]; assert flood_fill([[1]],0,0,1)==[[1]]",
37: "assert spiral_order([])==[]; assert spiral_order([[]])==[]; assert spiral_order([[1,2,3],[4,5,6]])==[1,2,3,6,5,4]",
38: "assert transpose([])==[]; assert transpose([[]])==[]; assert transpose([[1,2],[3,4]])==[[1,3],[2,4]]",
39: "assert k_largest([1,3,3,2],2)==[3,3]; assert k_largest([],0)==[]; assert k_largest([1],0)==[]",
40: "a=[3,1,2,1]; assert inversion_count(a)==4; assert a==[3,1,2,1]; assert inversion_count([])==0",
41: "assert parse_ints(' 1, -2 ')==[1,-2]; assert parse_ints(' ')==[]; raises(ValueError,lambda:parse_ints('1,,2'))",
42: "a=[1,2,3]; assert chunked(a,2)==[[1,2],[3]]; assert a==[1,2,3]; raises(ValueError,lambda:chunked(a,0))",
43: "assert flatten_once([1,[2,[3]],'ab'])==[1,2,[3],'ab']; assert flatten_once([])==[]",
44: "a=[{'id':1,'x':2},{'id':1,'x':3}]; assert dedupe_records(a,'id')==[a[0]]; raises(KeyError,lambda:dedupe_records([{}],'id'))",
45: "a=[{'name':'b','age':2},{'name':'a','age':2},{'name':'z','age':1}]; assert sort_records(a)==[a[2],a[1],a[0]]",
46: "assert word_counts('Hi, HI 123 a中B')=={'hi':2,'a':1,'b':1}; assert word_counts('')=={}",
47: "assert parse_query('a=1&a=&b=%E4%B8%AD')=={'a':['1',''],'b':['中']}; assert parse_query('')=={}",
48: "from urllib.parse import urlparse,parse_qs; u=build_url('https://example.com/x',{'q':'a 中','x':['1','2']}); assert parse_qs(urlparse(u).query)=={'q':['a 中'],'x':['1','2']}",
49: "import tempfile,pathlib,json; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name)/'x.json'; p.write_text('{\"x\":1}',encoding='utf-8'); assert read_json(p)=={'x':1}; p.write_text('{',encoding='utf-8'); raises(json.JSONDecodeError,lambda:read_json(p)); d.cleanup()",
50: "import tempfile,pathlib,json; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name)/'sub/x.json'; write_json(p,{'中文':1}); assert json.loads(p.read_text(encoding='utf-8'))=={'中文':1}; assert '中文' in p.read_text(encoding='utf-8'); d.cleanup()",
51: r"import tempfile,pathlib; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name)/'x.csv'; p.write_text('a,b\n\"x,y\",z\n',encoding='utf-8'); assert read_csv_rows(p)==[{'a':'x,y','b':'z'}]; d.cleanup()",
52: "import tempfile,pathlib,csv; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name)/'x.csv'; write_csv_rows(p,[{'a':'x,y','extra':1}],['a','b']); assert list(csv.DictReader(p.open(encoding='utf-8',newline='')))==[{'a':'x,y','b':''}]; d.cleanup()",
53: r"import tempfile,pathlib; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name)/'x'; p.write_text('  a  \n \n中\n',encoding='utf-8'); assert list(iter_nonempty_lines(p))==['  a  ','中']; d.cleanup()",
54: "assert count_extensions(['a.PY','b.py','README','.hidden','a.tar.gz'])=={'.py':2,'<none>':2,'.gz':1}",
55: r"from pathlib import Path; assert safe_child(Path('/tmp'),'x.txt')==Path('/tmp/x.txt'); raises(ValueError,lambda:safe_child('/tmp','../x')); raises(ValueError,lambda:safe_child('/tmp','a\\b')); raises(ValueError,lambda:safe_child('/tmp',''))",
56: "from datetime import date; assert parse_iso_date('2024-02-29')==date(2024,2,29); raises(ValueError,lambda:parse_iso_date('2023-02-29')); raises(ValueError,lambda:parse_iso_date('2024-2-9'))",
57: "from datetime import date; assert date_range(date(2024,2,28),date(2024,3,1))==[date(2024,2,28),date(2024,2,29),date(2024,3,1)]; assert date_range(date(2024,2,2),date(2024,2,1))==[]",
58: "a={'timestamp':'2024-01-01T00:30:00+02:00'}; r=group_by_day([a]); assert len(r)==1 and str(next(iter(r)))=='2023-12-31' and next(iter(r.values()))==[a]",
59: "assert format_duration(0)=='00:00:00'; assert format_duration(3661)=='01:01:01'; assert format_duration(90000)=='25:00:00'",
60: "from unittest.mock import Mock; f=Mock(side_effect=[TimeoutError(),3]); assert retry_call(f,2)==3 and f.call_count==2; raises(ValueError,lambda:retry_call(lambda:1,0)); raises(KeyError,lambda:retry_call(lambda:(_ for _ in ()).throw(KeyError()),2))",
61: "from unittest.mock import Mock; f=Mock(return_value=3); g=memoize(f); assert g(1)==g(1)==3 and f.call_count==1",
62: "r,t=timed_call(lambda x,y:x+y,2,y=3); assert r==5 and t>=0; raises(ValueError,lambda:timed_call(lambda:(_ for _ in ()).throw(ValueError())))",
63: "a={'TOKEN':'secret','name':'a','password':'x'}; assert redact_secrets(a)=={'TOKEN':'***','name':'a','password':'***'}; assert a['TOKEN']=='secret'",
64: "assert validate_config({'host':'localhost','port':80})=={'host':'localhost','port':80}; raises(ValueError,lambda:validate_config({'host':'x','port':True})); raises(ValueError,lambda:validate_config({'host':'','port':1}))",
65: "a={'x':1}; b={'x':None}; assert merge_defaults(a,b)=={'x':None}; assert a=={'x':1} and b=={'x':None}",
66: "a=append_item(1); b=append_item(2); assert a==[1] and b==[2]; c=[]; assert append_item(3,c) is c and c==[3]",
67: "assert search([],1)==-1; assert search([1],1)==0; assert search([1,3,5],4)==-1",
68: "a=[2,2,1,4,3]; assert remove_even(a)==[1,3]; assert a==[2,2,1,4,3]",
69: "a=make_grid(2,2); a[0][0]=1; assert a==[[1,0],[0,0]]; assert make_grid(0,2)==[]",
70: "assert [f(3) for f in multipliers()]==[0,3,6]",
71: "assert parse_number('12')==12; assert parse_number('x') is None; raises(TypeError,lambda:parse_number(None))",
72: "a=[2,1]; assert sorted_copy(a)==[1,2] and a==[2,1]",
73: "assert total_and_count(iter([1,2,3]))==(6,3); assert total_and_count(iter([]))==(0,0)",
74: "assert counts(['a','a','b'])=={'a':2,'b':1}; assert counts([])=={}",
75: "assert maximum([-3,-1])==-1; raises(ValueError,lambda:maximum([]))",
76: "assert is_quit(''.join(['qu','it'])); assert not is_quit('QUIT')",
77: "assert get_value({'x':0},'x',5)==0; assert get_value({'x':None},'x',5) is None; assert get_value({},'x',5)==5",
78: "a=[{'x':[1]}]; b=clone_records(a); b[0]['x'].append(2); assert a==[{'x':[1]}]",
79: "assert factorial(0)==1; assert factorial(5)==120; raises(ValueError,lambda:factorial(-1))",
80: "from decimal import Decimal; r=add_money('0.1','0.2'); assert isinstance(r,Decimal) and r==Decimal('0.3')",
81: "assert remove_txt('text.txt')=='text'; assert remove_txt('text')=='text'; assert remove_txt('a.txt.txt')=='a.txt'",
82: "assert common([1,2,2],[2,3])==[2]; assert common([],[1])==[]",
83: "a={'x':None,'y':0}; assert clean(a)=={'y':0} and a=={'x':None,'y':0}",
84: "assert inclusive_sum(1,3)==6; assert inclusive_sum(3,1)==0; assert inclusive_sum(-1,1)==0",
85: "assert valid_port(1); assert valid_port(65535); assert not valid_port(True); assert not valid_port(0)",
}

HELPER = '''
def _pyforge_raises(kind, func):
    try:
        func()
    except kind:
        return
    raise AssertionError('expected exception')
class Solution:
    def verify(self):
        raises = _pyforge_raises
        {checks}
        return True
'''


def implementation_only(code):
    """移除教师额外附加的 pytest 测试，不改变实现函数体。"""
    tree = ast.parse(code)
    body = []
    changed = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [name for name in node.names if name.name != "pytest"]
            if len(names) != len(node.names):
                changed = True
                if names:
                    body.append(ast.Import(names=names))
                continue
        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            changed = True
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fixture = any("fixture" in ast.unparse(decorator) for decorator in node.decorator_list)
            if node.name.startswith("test_") or node.name.startswith("Test") or fixture:
                changed = True
                continue
        body.append(node)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))) if changed else code


def verify(source, output, image="python:3.12-slim"):
    source, output = Path(source), Path(output)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "verification_manifest.json"
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "checks_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "image_id": docker_environment(image)}
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("审核配置已变化，请使用新输出目录")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    journal = output / "verification.jsonl"
    existing = {r["id"]:r for r in map(json.loads,journal.read_text(encoding="utf-8").splitlines())} if journal.exists() else {}
    results = []
    for row in rows:
        if row["id"] in existing:
            row.update(existing[row["id"]])
            results.append(row)
            continue
        number = int(row["id"].split("-")[-1])
        blocks = re.findall(r"```(?:python|python3|py)\s*\n(.*?)```", row["messages"][-1]["content"], re.S)
        check = CHECKS.get(number)
        if check and blocks:
            code = implementation_only(blocks[0])
            result = run_isolated(code + "\n" + HELPER.replace("{checks}", check),
                {"method":"verify","comparator":"exact","tests":[{"args":[],"expected":True}]},manifest["image_id"],timeout=10)
            row["verification"] = {"syntax":True,"execution":result["status"]=="passed","human_review":False,"execution_status":result["status"]}
            if result["status"] == "passed":
                # 保留原始响应文件；训练候选移除附加测试代码块，避免导入不存在的文件。
                original = row["messages"][-1]["content"]
                first = re.search(r"```(?:python|python3|py)\s*\n.*?```",original,re.S)
                row["messages"][-1]["content"] = original[:first.start()] + "```python\n" + code.rstrip() + "\n```"
                row["normalization"] = "保留首个完整实现及前置说明；移除额外 pytest 测试、夹具与其导入；原始响应单独保留"
        row["status"] = "execution_checked" if row["verification"]["execution"] else "pending_review"
        event = {k:row[k] for k in ("id","verification","status","messages")}
        if "normalization" in row:
            event["normalization"] = row["normalization"]
        with journal.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+"\n")
        results.append(row)
        print(row["id"],row["verification"].get("execution_status","manual_review_needed"),flush=True)
    for split in ("train","validation","test"):
        selected=[row for row in results if row["split"]==split and row["verification"]["execution"]]
        (output/(split+".jsonl")).write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in selected),encoding="utf-8")
    pending=[r for r in results if not r["verification"]["execution"]]
    (output/"pending_review.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in pending),encoding="utf-8")
    summary = {"verified":len(results)-len(pending),"pending":len(pending),
        "splits":{split:sum(r["split"]==split and r["verification"]["execution"] for r in results) for split in ("train","validation","test")},
        "files_sha256":{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ("train.jsonl","validation.jsonl","test.jsonl","pending_review.jsonl")},
        "limitations":"仅通过自有功能用例，不代表完整正确性；解释与测试生成题仍待审核"}
    (output/"dataset_manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default="outputs/teacher_v1/candidates.jsonl")
    parser.add_argument("--output",default="outputs/legacy_teacher_review")
    args=parser.parse_args()
    print(json.dumps(verify(args.source,args.output),ensure_ascii=False,indent=2))
