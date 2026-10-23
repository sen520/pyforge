"""生成原创种子题；不调用 API，不生成答案，不作为已验证训练集。"""

import argparse
import json
from pathlib import Path


ALGORITHMS = """
two_sum(nums, target)|返回两个不同下标使对应整数之和为 target；无解返回 None，多解返回任一对
unique_sorted(nums)|返回整数列表去重后的升序列表，不修改输入
merge_sorted(a, b)|合并两个已升序整数列表，保留重复值，时间复杂度 O(n+m)
binary_search(nums, target)|在升序整数列表查找 target，返回最左下标，无解返回 -1，复杂度 O(log n)
lower_bound(nums, target)|返回升序整数列表中第一个不小于 target 的位置，若不存在返回列表长度
rotate_right(nums, k)|返回向右循环移动 k 位的新列表，k 为非负整数，空列表返回空列表
move_zeros(nums)|返回将所有整数 0 移到末尾的新列表，非零元素相对顺序不变
max_subarray(nums)|返回非空整数列表的最大连续子数组之和，空列表抛 ValueError
longest_increasing_run(nums)|返回严格递增连续区间的最大长度，空列表返回 0
product_except_self(nums)|返回每个位置以外元素的乘积，禁止除法，空列表返回空列表，单元素返回 [1]
merge_intervals(intervals)|合并闭区间，端点为整数且 start<=end；接触端点也合并，输出按起点排序
intersection(a, b)|返回两个整数列表的去重交集，按升序排列
first_unique(text)|返回字符串首个只出现一次字符的下标，无解返回 -1，区分大小写
is_palindrome(text)|忽略非 ASCII 字母数字字符及大小写判断回文，过滤后为空返回 True
anagram_groups(words)|按字符频次分组小写英文单词，组按首次出现顺序，组内保留输入顺序
longest_unique_substring(text)|返回无重复字符的最长连续子串长度，按 Unicode 字符处理
valid_brackets(text)|判断仅含 ()[]{} 的字符串括号是否匹配，空串返回 True，其他字符抛 ValueError
run_length_encode(text)|返回连续字符的 [字符, 次数] 列表，不拼接成字符串，空串返回空列表
substring_index(text, pattern)|返回模式串首次出现的位置，无解返回 -1，空模式返回 0，禁止 find/index
common_prefix(words)|返回字符串列表的最长公共前缀，空列表返回空串
gcd(a, b)|用欧几里得算法返回两个整数的非负最大公约数，gcd(0,0)=0，禁止 math.gcd
is_prime(n)|判断整数是否为素数，n<2 返回 False，试除上界不超过平方根
sieve(n)|返回所有不大于非负整数 n 的素数，使用埃拉托斯特尼筛法
fast_power(base, exponent)|指数为非负整数，用快速幂计算整数幂，约定 0 的 0 次幂为 1
popcount(n)|返回非负整数二进制表示中 1 的个数，禁止 bit_count 与字符串转换
fibonacci(n)|返回第 n 个斐波那契数，F0=0,F1=1，n 为非负整数，额外空间 O(1)
climb_stairs(n)|每次走 1 或 2 级，返回到达非负整数 n 级台阶的方法数，n=0 返回 1
min_coins(coins, amount)|硬币面额为不同正整数且数量无限，返回组成非负 amount 的最少枚数，无解返回 -1
knapsack(weights, values, capacity)|0/1 背包最大价值，重量为正整数、价值非负、长度相等、容量非负，允许不选
longest_common_subsequence(a, b)|返回两个字符串最长公共子序列长度，空字符串应正确处理
edit_distance(a, b)|返回插入、删除、替换各代价 1 的最小编辑距离
grid_paths(rows, cols)|仅向右或向下移动，返回正整数行列网格左上到右下路径数；任一维为 0 返回 0
shortest_path(graph, start, goal)|无权有向图邻接字典，返回最短路径节点列表，不可达返回 None，缺失节点视为无出边
connected_components(n, edges)|编号 0 到 n-1 的无向图，返回连通分量数，允许自环及重复边，n=0 返回 0
topological_order(n, edges)|有向边 u 到 v，返回任一拓扑序，有环返回 None；节点为 0 到 n-1，边无重复
flood_fill(image, row, col, color)|矩形整数矩阵按四邻域填充，返回新矩阵，不修改输入，坐标合法
spiral_order(matrix)|按顺时针返回矩形整数矩阵元素，空矩阵或零列返回空列表
transpose(matrix)|返回矩形矩阵转置，空矩阵或零列返回空列表，输入不可修改
k_largest(nums, k)|返回整数列表最大的 k 个值，保留重复并降序排列，0<=k<=len(nums)，用大小为 k 的堆
inversion_count(nums)|统计 i<j 且 nums[i]>nums[j] 的对数，使用 O(n log n) 算法，不修改输入
"""

PRACTICAL = """
parse_ints(text)|将逗号分隔文本解析为整数列表，允许两侧空白，空白输入返回 []，空字段或非法整数抛 ValueError
chunked(items, size)|将列表分为长度至多 size 的新列表块，size 为正整数，否则抛 ValueError
flatten_once(items)|只展开一层列表元素，其他元素原样保留，不递归，不修改输入
dedupe_records(records, key)|按字典指定键的可哈希值去重，保留第一次出现，缺键抛 KeyError
sort_records(records)|字典含 name 字符串与 age 整数，先 age 升序再 name 升序，返回新列表
word_counts(text)|将 ASCII 英文字母连续片段视为单词，忽略大小写，返回频数字典
parse_query(query)|解析 URL 查询串为 key 到字符串列表的字典，保留空值与重复键，使用 urllib.parse
build_url(base, params)|base 不含查询和片段，params 值为字符串或字符串列表，正确编码空格和 Unicode
read_json(path)|以 UTF-8 读取 JSON，文件缺失与无效 JSON 保留标准库原始异常
write_json(path, value)|以 UTF-8 写 JSON，保留中文、缩进 2，父目录不存在时创建，不能执行输入字符串
read_csv_rows(path)|用 csv.DictReader 读取 UTF-8 CSV，返回字典列表，正确处理引号内逗号和换行
write_csv_rows(path, rows, fields)|按 fields 列顺序写 UTF-8 CSV，缺字段写空串，多余字段忽略，正确设置 newline
iter_nonempty_lines(path)|生成器逐行读取 UTF-8 文件，去掉行尾换行但保留其他空格，跳过仅含空白的行
count_extensions(paths)|输入文件名字符串列表，以 pathlib 获取最后扩展名并转小写，无扩展名记为 <none>，返回频数
safe_child(root, name)|仅接受单层文件名，拒绝空串、点目录、正反斜杠、绝对路径，返回 root 下 Path；不访问文件系统
parse_iso_date(text)|解析严格 YYYY-MM-DD 日期返回 date，格式或日期非法抛 ValueError
date_range(start, end)|输入 date 对象，返回含起止的日期列表，start>end 返回 []
group_by_day(events)|事件字典含带时区的 ISO 时间字符串 timestamp，转换到 UTC 按日期分组，保留原始事件顺序
format_duration(seconds)|非负整数秒转为 HH:MM:SS，小时允许超过 24，至少两位
retry_call(func, attempts)|仅对 TimeoutError 重试，attempts 为正整数，成功立即返回，耗尽抛最后异常，不等待
memoize(func)|实现仅适用于位置参数可哈希的函数缓存装饰器，使用 wraps，其他异常不缓存
timed_call(func, *args, **kwargs)|返回 (结果, 耗时秒数)，用 perf_counter，异常直接传播
redact_secrets(record)|浅复制字典，将不区分大小写的 password、token、api_key 对应值替换为 ***，不修改输入
validate_config(config)|要求 host 为非空字符串、port 为 1..65535 整数且不接受 bool，返回规范字典；非法抛 ValueError
merge_defaults(defaults, overrides)|浅合并返回新字典，overrides 显式 None 仍覆盖默认值，不修改输入
"""

BUGS = [
    ("共享默认列表", "def append_item(item, items=[]):\n    items.append(item)\n    return items", "省略 items 时每次调用独立；显式传入列表时应原地追加并返回该列表"),
    ("二分边界", "def search(nums, target):\n    left, right = 0, len(nums)\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1", "升序列表查找任一匹配下标，无解返回 -1，空列表可用"),
    ("遍历时删除", "def remove_even(nums):\n    for x in nums:\n        if x % 2 == 0:\n            nums.remove(x)\n    return nums", "返回只含奇数的新列表，保持顺序，不修改输入"),
    ("列表别名", "def make_grid(rows, cols):\n    return [[0] * cols] * rows", "非负行列创建矩阵，不同行互不共享列表"),
    ("闭包捕获", "def multipliers():\n    return [lambda x: i * x for i in range(3)]", "返回三个函数，分别对输入乘以 0、1、2"),
    ("异常类型", "def parse_number(text):\n    try:\n        return int(text)\n    except KeyError:\n        return None", "字符串无法解析整数时返回 None，其他非预期异常不吞掉"),
    ("排序返回值", "def sorted_copy(nums):\n    return nums.sort()", "返回升序新列表，不修改输入"),
    ("生成器耗尽", "def total_and_count(values):\n    return sum(values), len(list(values))", "输入任意有限整数迭代器，单次遍历返回总和与数量"),
    ("计数覆盖", "def counts(items):\n    result = {}\n    for item in items:\n        result[item] = 1\n    return result", "可哈希元素频次统计"),
    ("负数最大值", "def maximum(nums):\n    best = 0\n    for n in nums:\n        best = max(best, n)\n    return best", "非空整数列表最大值，空列表抛 ValueError"),
    ("字符串身份比较", "def is_quit(command):\n    return command is 'quit'", "按字符串内容严格等于 quit 判断，不忽略大小写或空白"),
    ("零值丢失", "def get_value(data, key, default):\n    return data.get(key) or default", "仅缺键时返回默认值，0、False、空串及 None 都保留"),
    ("浅拷贝嵌套对象", "def clone_records(records):\n    return records.copy()", "由 JSON 可表示值组成的嵌套列表和字典独立复制"),
    ("缺少返回", "def factorial(n):\n    if n == 0:\n        return 1\n    n * factorial(n - 1)", "非负整数阶乘，负数抛 ValueError"),
    ("浮点金额", "def add_money(a, b):\n    return float(a) + float(b)", "输入十进制金额字符串，用 Decimal 精确相加并返回 Decimal，不额外舍入"),
    ("移除后缀", "def remove_txt(name):\n    return name.rstrip('.txt')", "只移除一个完全匹配的小写 .txt 后缀，其他字符原样保留"),
    ("集合交集错误", "def common(a, b):\n    return sorted(set(a) or set(b))", "返回两个整数列表的去重升序交集"),
    ("字典迭代修改", "def clean(data):\n    for key in data:\n        if data[key] is None:\n            del data[key]\n    return data", "返回去除 None 值的新字典，不修改输入"),
    ("端点遗漏", "def inclusive_sum(start, end):\n    return sum(range(start, end))", "返回含两个端点的整数区间总和，start>end 返回 0"),
    ("类型校验接受布尔", "def valid_port(port):\n    return isinstance(port, int) and 1 <= port <= 65535", "仅接受 1..65535 的整数端口，True 和 False 均非法"),
]

REVIEWS = r"""
test|为 clamp(value, low, high) 编写 pytest 测试：上下界为整数且 low<=high，返回夹在闭区间的整数，low>high 抛 ValueError
test|为 parse_ints(text) 编写 pytest 测试：逗号分隔整数，允许周围空白，空白文本返回 []，空字段和非法整数抛 ValueError
test|为 chunked(items, size) 编写 pytest 测试：正整数 size 分块，非法 size 抛 ValueError，输入不得被修改
test|为 merge_intervals(intervals) 编写 pytest 测试：合并闭区间，接触端点也合并，返回按起点排序的新列表
test|为 first_unique(text) 编写 pytest 测试：首个仅出现一次字符下标，无解返回 -1，区分大小写，支持 Unicode
test|为 retry_call(func, attempts) 编写 pytest 测试：仅重试 TimeoutError，次数必须正整数，成功立即返回；使用 mock 验证调用次数
test|为 write_json(path, value) 编写 pytest 测试：UTF-8、保留中文、创建父目录；使用 tmp_path，验证文件可被 json 解析
test|为 dedupe_records(records, key) 编写 pytest 测试：按键值保留首次记录，缺键抛 KeyError，输入不修改
explain|解释这段代码的结果、原因及两个独立行的创建方法：rows = [[0] * 2] * 2; rows[0][0] = 7; print(rows)
explain|解释生成器为何只遍历一次，以及如何单次计算总和与数量：it = iter([1, 2, 3]); print(sum(it)); print(list(it))
explain|解释闭包延迟绑定并修复：functions = [lambda: i for i in range(3)]; print([f() for f in functions])
explain|解释浅拷贝后修改嵌套列表的影响：a = {'x': [1]}; b = a.copy(); b['x'].append(2); print(a)
explain|解释 finally 对返回值的影响，并给出不覆盖正常结果的改法：def f():\n    try:\n        return 1\n    finally:\n        return 2
explain|解释可变默认参数的问题：def f(x, items=[]):\n    items.append(x)\n    return items；给出连续两次调用结果与修复
explain|比较 sorted(values) 和 values.sort() 的返回值、原地修改与稳定性，用可运行小例子展示
"""


def entries(text):
    return [line.strip().split("|", 1) for line in text.strip().splitlines()]


def questions():
    result = []
    for category, source in (("algorithm", ALGORITHMS), ("practical", PRACTICAL)):
        for signature, requirement in entries(source):
            result.append((category, signature, "请实现 {}。要求：{}。".format(signature, requirement)))
    for title, code, requirement in BUGS:
        result.append(("debugging", title, "修复以下 Python 代码，说明缺陷并提供完整修复代码。目标：{}。\n```python\n{}\n```".format(requirement, code)))
    for category, prompt in entries(REVIEWS):
        result.append(("test_generation" if category == "test" else "explanation", prompt.split("：")[0], prompt.replace("\\n", "\n")))
    # 同一知识任务的实现、修复、测试与解释必须共享分组，避免后续划分泄漏。
    related = {
        67: 4, 75: 8, 77: 65, 82: 12, 85: 64,
        87: 41, 88: 42, 89: 11, 90: 13, 91: 60, 92: 50, 93: 44,
        94: 69, 95: 73, 96: 70, 97: 78, 99: 66, 100: 72,
    }
    # 后续发现更多语义近似题时，审核者仍需合并 group_id。
    for index, (category, title, prompt) in enumerate(result, 1):
        family = index
        while family in related:
            family = related[family]
        yield {"id": "seed-{:03d}".format(index), "task_type": category, "title": title,
               "language": "zh", "source": "pyforge_original_seed_v1", "license": "project_original_pending_license",
               "group_id": "seed-family-{:03d}".format(family), "python_version": "3.12",
               "prompt": prompt, "answer_requirements": "使用 Python 标准库（测试题可用 pytest）；给出简短说明及完整代码；覆盖边界情况。算法题注明复杂度。不要输出冗长推理过程。",
               "status": "draft_requires_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    rows = list(questions())
    assert len(rows) == 100
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("已生成 {} 道待审核种子题：{}".format(len(rows), path))


if __name__ == "__main__":
    main()
