"""从本项目直接编写的答案构建数据，不读取 .env、不调用 API。"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from pyforge.data import split_records
from codex_review_answers import EXPLANATIONS, TESTS


ALGORITHM_NOTES = '''
用哈希表记录此前元素的下标，先查补数再写入，避免重复使用同一位置。期望时间 O(n)，空间 O(n)。
先用集合去重，再升序排序；不修改输入。时间 O(n log n)，空间 O(n)。
用双指针依次取较小元素，再追加未处理的尾部。时间 O(n+m)，输出空间 O(n+m)。
先找第一个不小于 target 的位置，再确认相等，保证返回最左下标。时间 O(log n)，空间 O(1)。
在左闭右开的区间二分，退出时 left 就是插入位置。时间 O(log n)，空间 O(1)。
先处理空列表，再将位移对长度取模，用切片构造新列表。时间和空间均为 O(n)。
保留非零元素的相对顺序，再补上同样数量的零。时间 O(n)，输出空间 O(n)。
维护以当前位置结尾的最大子数组和，以及全局最大值；不能用 0 初始化而忽略全负输入。时间 O(n)，额外空间 O(1)。
遇到非严格递增的位置就从长度 1 重新计数。时间 O(n)，额外空间 O(1)。
分别累计前缀积和后缀积，天然支持零，不需要除法。时间 O(n)，除输出外空间 O(1)。
排序后仅当下一闭区间起点不大于当前终点时合并。[1,2] 与 [3,4] 不接触，不能合并；不修改输入。时间 O(n log n)，空间 O(n)。
用集合交集去重，再排序。设两输入长度为 n、m，交集大小为 k，期望时间 O(n+m+k log k)，空间 O(n+m)。
先统计频次，再按原顺序找首个频次为 1 的字符。按 Unicode 字符处理，时间 O(n)，空间 O(k)，k 为不同字符数。
只保留 ASCII 字母数字并转为小写，再比较反转结果。时间和空间均为 O(n)。
以 26 个小写字母的频次元组为键，字典保留首次插入顺序。设总字符数为 L、单词数为 n，时间 O(L+n)，空间 O(L+n)。
滑动窗口的左端只向右移动，跳过重复字符的上次位置。时间 O(n)，空间 O(k)，k 为不同字符数。
先校验字符，再用栈匹配左右括号。空串匹配成功。时间 O(n)，空间 O(n)。
逐个字符合并连续段，输出字符和次数的二元素列表。时间 O(n)，空间 O(n)。
用 KMP 前缀表避免重复比较，不调用 find 或 index。文本长 n、模式长 m，时间 O(n+m)，空间 O(m)。
逐列比较，首个不一致处停止。最坏时间 O(nm)，额外空间 O(n+m)，n 为单词数、m 为最短长度。
先取绝对值，再反复执行余数递推。时间 O(log(max(|a|,|b|)+1))，额外空间 O(1)，按算术操作计数。
小于 2 的数不是素数，只试除到整数平方根。时间 O(sqrt(n))，额外空间 O(1)。
从每个素数的平方开始标记合数。时间 O(n log log n)，空间 O(n)。
按二进制拆分指数，反复平方；初始结果 1 也覆盖 0 的 0 次幂。算术操作次数 O(log(exponent+1))，额外标量空间 O(1)。
每次用 n & (n-1) 清除最低位的 1。时间 O(k)，k 为置位数量，额外标量空间 O(1)。
迭代维护相邻两个斐波那契数。执行 O(n) 次加法，只保存 O(1) 个整数；大整数的实际位数随 n 增长。
状态表示到达相邻两级的走法数，0 级按一种空走法计。执行 O(n) 次加法，只保存 O(1) 个整数。
dp[x] 记录凑出 x 的最少枚数，用大于 amount 的值表示不可达。硬币种类为 c，时间 O(c*amount)，空间 O(amount)。
容量从大到小更新，防止一件物品被重复使用。物品数 n、容量 C，时间 O(nC)，空间 O(C)。
滚动一行 DP，并保存更新前的左上角状态。时间 O(len(a)*len(b))，空间 O(len(b))。
插入、删除、替换分别对应左、上、左上状态；相同字符直接沿用左上状态。时间 O(len(a)*len(b))，空间 O(len(b))。
一维 DP 累加上方与左方路径数，零维网格直接返回 0。时间 O(rows*cols)，空间 O(cols)。
用 BFS 找最短路，并保存父节点重建路径；缺失节点没有出边。时间 O(V+E)，空间 O(V)。
构造无向邻接表，逐个未访问节点做迭代 DFS。时间和空间均为 O(n+E)，重复边和自环不增加分量数。
使用入度与队列进行 Kahn 拓扑排序；输出不足 n 个节点说明存在环。时间和空间均为 O(n+E)。
先复制矩阵，再迭代填充四邻域；新旧颜色相同时直接返回副本。时间和空间均为 O(rows*cols)。
维护四条边界，转弯后检查剩余行列，避免重复访问单行或单列。时间 O(rows*cols)，除输出外空间 O(1)。
用 zip 按列收集，再转换成新的行列表。时间和输出空间均为 O(rows*cols)。
维护最多 k 个元素的小顶堆，最后对堆降序排序。时间 O(n log(k+1)+k log(k+1))，空间 O(k)。
在归并时统计右元素跨过的左侧剩余元素数；相等元素不计逆序。时间 O(n log n)，空间 O(n)，不修改输入。
'''.strip().splitlines()

PRACTICAL_NOTES = '''
空白文本返回空列表；其他情况逐字段调用 int，空字段和非法整数字符串会自然抛 ValueError。
先检查 size 必须为正整数，再用切片构造新块，不修改输入。
仅将列表元素展开一层，字符串等其他对象作为单个元素保留。
用集合记录已经出现的键值，按原顺序保留首条记录；直接索引使缺键自然抛 KeyError。
sorted 返回新列表，复合排序键依次为 age 和 name。
先在原文本中提取 ASCII 字母片段，再忽略大小写计数，避免 Unicode 大小写转换改变分词边界。
使用 parse_qs 并开启 keep_blank_values，重复键保留为列表，百分号编码自动解码。
使用 urlencode 的 doseq 参数展开列表值，自动处理 Unicode 和空格。
用 UTF-8 和上下文管理器读取；不吞掉文件缺失或 JSON 解析异常。
先创建父目录，再用 ensure_ascii=False、indent=2 写入 UTF-8 JSON，不执行输入内容。
csv.DictReader 负责带引号逗号和跨行字段；newline='' 让 csv 模块处理换行。
固定列顺序并忽略额外字段，DictWriter 默认将缺失字段写为空串。
生成器逐行读取，只移除行尾换行，跳过仅含空白的行，保留其他空格。
只取最后一个扩展名并转小写；隐藏文件名本身没有扩展名时按 <none> 计数。
拒绝分隔符、点目录和 Windows 盘符等路径结构，仅进行字符串与路径解析，不访问文件系统。
先严格校验 ASCII YYYY-MM-DD 外形，再由 date.fromisoformat 验证闰年和实际日期。
利用日期差构造含起止的日期列表，反向区间的 range 为空。
拒绝无时区时间，转换到 UTC 后取日期分组，同日事件保持输入顺序。
用 divmod 分解时、分、秒，小时不对 24 取模。
只捕获 TimeoutError；重试耗尽使用裸 raise 保留最后一个异常，其他异常立即传播。
用位置参数元组作缓存键，wraps 保留函数元数据；只有调用成功才写入缓存。
使用 perf_counter 测量耗时，函数异常自然传播，不伪造返回结果。
创建新字典，仅将指定敏感字段的值替换为 ***；大小写不敏感，输入不变。
显式使用 type(port) is int 排除 bool，再检查端口范围和非空主机字符串。
创建新字典，右侧覆盖左侧，显式 None 也是有效覆盖值。
'''.strip().splitlines()

BUG_NOTES = '''
默认列表在函数定义时只创建一次，应改用 None，并且只在省略列表时新建对象。
右边界必须是最后一个有效下标，而不是列表长度；空列表自然跳过循环。
遍历时删除会跳过相邻元素，也会修改输入；改为列表推导式创建结果。
外层列表乘法会共享内层列表引用；用推导式逐行创建。
闭包在调用时读取循环变量；用默认参数绑定每次循环的值。
int 对非法字符串抛 ValueError，应只捕获该异常，TypeError 等其他异常保留。
list.sort 原地修改且返回 None；这里需要 sorted 返回新列表。
输入可能是一次性迭代器；在一次循环中同时累计总和与数量。
不能将每次出现都写为 1，应累计频次。
用 0 初始化会错误处理全负数；显式处理空输入，再返回真实最大值。
is 比较对象身份，== 才比较字符串内容。
or 会覆盖 0、False、空串和 None，dict.get 的默认参数只在缺键时生效。
浅复制仍共享嵌套对象，使用 deepcopy 复制嵌套列表与字典。
原递归遗漏 return；迭代计算避免递归深度限制，负数主动报错。
二进制浮点无法精确表达很多十进制小数。直接从字符串构造 Decimal，并为输入长度预留足够精度。
rstrip 的参数是字符集合，不是完整后缀；用 removesuffix 只移除一次匹配后缀。
or 是逻辑运算，不是集合交集；使用 & 后升序排序。
不能在迭代原字典时删除键；用推导式返回新字典。
range 的终点不包含在内；用等差数列公式计算含端点总和，反向区间返回 0。
bool 是 int 的子类，使用精确类型检查排除布尔值。
'''.strip().splitlines()


def authored_rows(source):
    questions = [json.loads(line) for line in Path(source).read_text(encoding='utf-8').splitlines()]
    if [q['id'] for q in questions] != [f'seed-{i:03d}' for i in range(1,101)]:
        raise ValueError('需要原始有序 100 题，防止答案错配')
    path = Path(__file__).with_name('codex_solutions.py')
    text = path.read_text(encoding='utf-8')
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 85:
        raise ValueError('实现数量不正确')
    notes = ALGORITHM_NOTES + PRACTICAL_NOTES + BUG_NOTES
    assert len(notes) == 85
    assignments = {q['id']: name for name, rows in split_records(questions).items() for q in rows}
    rows = []
    for number, question in enumerate(questions,1):
        if number <= 85:
            node = functions[number-1]
            if number <= 65 and question['title'].split('(')[0] != node.name:
                raise ValueError('题目与函数不匹配：'+question['id'])
            code = ast.get_source_segment(text,node)
            explanation = notes[number-1]
        elif number <= 93:
            code = TESTS[number]
            explanation = '将被测实现放在 solution.py，下面保存为 test_solution.py，运行 python -m pytest。测试覆盖正常输入、边界条件及题目约定的异常，不依赖网络。'
        else:
            explanation, code = EXPLANATIONS[number]
        compile(code,question['id'],'exec')
        row = {key:question[key] for key in ('id','task_type','language','license','group_id')}
        row.update(source='codex_authored_v1', author='Codex', authorship='directly_written_without_external_model_api',
            split=assignments[question['id']], status='authored_pending_execution',
            verification={'syntax':True,'execution':False,'human_review':False},
            messages=[{'role':'system','content':'你是 Python 编程助手。遵守接口与约束，用简短说明和完整代码回答。'},
                      {'role':'user','content':question['prompt']},
                      {'role':'assistant','content':explanation+'\n\n```python\n'+code.rstrip()+'\n```'}])
        rows.append(row)
    return rows


def build(source, output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空，请使用新版本路径')
    rows=authored_rows(source)
    output.mkdir(parents=True,exist_ok=True)
    path=output/'candidates.jsonl'
    path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8',newline='\n')
    manifest={'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'candidates_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':100,
        'external_model_api_calls':0,'author':'Codex',
        'authoring_files_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('codex_solutions.py','codex_review_answers.py','build_codex_data.py')}}
    (output/'authorship_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='datasets/seed_questions.jsonl')
    parser.add_argument('--output',default='data/codex_v1')
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output),ensure_ascii=False,indent=2))
