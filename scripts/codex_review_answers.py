"""Codex 编写的测试题与解释题答案，独立于历史教师响应。"""

TESTS = {
86: '''import pytest
from solution import clamp

@pytest.mark.parametrize('value,low,high,expected', [
    (5, 0, 10, 5), (-1, 0, 10, 0), (11, 0, 10, 10),
    (0, 0, 10, 0), (10, 0, 10, 10), (8, 3, 3, 3), (-5, -4, -2, -4),
])
def test_clamp(value, low, high, expected):
    assert clamp(value, low, high) == expected

def test_reversed_bounds():
    with pytest.raises(ValueError):
        clamp(0, 2, 1)
''',
87: '''import pytest
from solution import parse_ints

@pytest.mark.parametrize('text,expected', [
    ('', []), ('  ', []), ('1', [1]), (' 1, -2, +3 ', [1, -2, 3]), ('0,0', [0, 0]),
])
def test_valid(text, expected):
    assert parse_ints(text) == expected

@pytest.mark.parametrize('text', ['1,,2', ',1', '1,', 'x', '1.5', '1,   ,2'])
def test_invalid(text):
    with pytest.raises(ValueError):
        parse_ints(text)
''',
88: '''import pytest
from solution import chunked

@pytest.mark.parametrize('items,size,expected', [
    ([], 2, []), ([1], 3, [[1]]), ([1,2,3], 2, [[1,2],[3]]),
    ([1,2], 1, [[1],[2]]), ([1,2], 2, [[1,2]]),
])
def test_chunks(items, size, expected):
    original = items[:]
    result = chunked(items, size)
    assert result == expected
    assert items == original
    if result:
        result[0].append('new')
        assert items == original

@pytest.mark.parametrize('size', [0, -1, 1.5, True, '2'])
def test_bad_size(size):
    with pytest.raises(ValueError):
        chunked([1], size)
''',
89: '''import pytest
from solution import merge_intervals

@pytest.mark.parametrize('intervals,expected', [
    ([], []), ([[2,2]], [[2,2]]), ([[3,4],[1,3]], [[1,4]]),
    ([[1,2],[3,4]], [[1,2],[3,4]]), ([[1,8],[2,3],[8,9]], [[1,9]]),
    ([[5,6],[1,2],[1,2]], [[1,2],[5,6]]),
])
def test_merge(intervals, expected):
    assert [list(pair) for pair in merge_intervals(intervals)] == expected
''',
90: '''import pytest
from solution import first_unique

@pytest.mark.parametrize('text,expected', [
    ('', -1), ('a', 0), ('aabb', -1), ('aabcb', 3),
    ('AaA', 1), ('中a中b', 1), ('🙂🙂x', 2),
])
def test_first_unique(text, expected):
    assert first_unique(text) == expected
''',
91: '''import pytest
from unittest.mock import Mock
from solution import retry_call

def test_success_immediately():
    func = Mock(return_value=7)
    assert retry_call(func, 3) == 7
    func.assert_called_once_with()

def test_retry_then_success():
    func = Mock(side_effect=[TimeoutError(), TimeoutError(), 9])
    assert retry_call(func, 3) == 9
    assert func.call_count == 3

def test_exhaustion_preserves_last_exception():
    last = TimeoutError('last')
    func = Mock(side_effect=[TimeoutError('first'), last])
    with pytest.raises(TimeoutError) as caught:
        retry_call(func, 2)
    assert caught.value is last
    assert func.call_count == 2

def test_other_exception_is_not_retried():
    func = Mock(side_effect=KeyError('bad'))
    with pytest.raises(KeyError):
        retry_call(func, 3)
    func.assert_called_once_with()

@pytest.mark.parametrize('attempts', [0, -1, True, 1.5])
def test_invalid_attempts(attempts):
    with pytest.raises(ValueError):
        retry_call(lambda: 1, attempts)
''',
92: '''import json
from solution import write_json

def test_utf8_and_parent_directory(tmp_path):
    path = tmp_path / 'new' / 'nested' / 'data.json'
    value = {'中文': ['你好', 1, None]}
    write_json(path, value)
    text = path.read_text(encoding='utf-8')
    assert json.loads(text) == value
    assert '中文' in text and '你好' in text
    assert text == json.dumps(value, ensure_ascii=False, indent=2)

def test_replace_existing_file(tmp_path):
    path = tmp_path / 'data.json'
    write_json(path, {'old': 1})
    write_json(path, [])
    assert json.loads(path.read_text(encoding='utf-8')) == []
''',
93: '''import copy
import pytest
from solution import dedupe_records

def test_first_record_wins_and_input_unchanged():
    rows = [{'id': 1, 'value': 'first'}, {'id': 1, 'value': 'later'}, {'id': 2}]
    original = copy.deepcopy(rows)
    assert dedupe_records(rows, 'id') == [rows[0], rows[2]]
    assert rows == original

def test_empty():
    assert dedupe_records([], 'id') == []

def test_missing_key():
    with pytest.raises(KeyError):
        dedupe_records([{'id': 1}, {}], 'id')

def test_falsy_keys_are_valid():
    rows = [{'id': None, 'x': 1}, {'id': None, 'x': 2}, {'id': 0}]
    assert dedupe_records(rows, 'id') == [rows[0], rows[2]]
''',
}

EXPLANATIONS = {
94: ("输出 [[7, 0], [7, 0]]。外层列表保存了同一个内层列表的两次引用，修改任一行会同时反映到另一行。用列表推导式分别创建每行。", '''rows = [[0] * 2] * 2
rows[0][0] = 7
assert rows == [[7, 0], [7, 0]]
independent = [[0] * 2 for _ in range(2)]
independent[0][0] = 7
assert independent == [[7, 0], [0, 0]]
assert independent[0] is not independent[1]
'''),
95: ("依次输出 6 和 []。it 是有状态的迭代器，sum 已将其耗尽，之后不会自动回到开头。若需要总和与数量，应在一次循环中同时累计；空迭代器返回 (0, 0)。", '''def total_and_count(values):
    total = count = 0
    for value in values:
        total += value
        count += 1
    return total, count

it = iter([1, 2, 3])
assert sum(it) == 6
assert list(it) == []
assert total_and_count(iter([1, 2, 3])) == (6, 3)
assert total_and_count(iter([])) == (0, 0)
'''),
96: ("原代码输出 [2, 2, 2]。闭包在调用时读取同一个循环变量 i，而此时循环已结束，i 为 2。用默认参数在每次创建函数时绑定当前值。", '''functions = [lambda: i for i in range(3)]
assert [f() for f in functions] == [2, 2, 2]
fixed = [lambda value=i: value for i in range(3)]
assert [f() for f in fixed] == [0, 1, 2]
'''),
97: ("输出 {'x': [1, 2]}。dict.copy 仅复制最外层字典，a 和 b 的 x 仍引用同一个列表。需要独立嵌套数据时使用 deepcopy。", '''from copy import deepcopy
a = {'x': [1]}
b = a.copy()
b['x'].append(2)
assert a == {'x': [1, 2]}
assert a is not b and a['x'] is b['x']
c = deepcopy(a)
c['x'].append(3)
assert a == {'x': [1, 2]}
assert c == {'x': [1, 2, 3]}
'''),
98: ("原函数返回 2：finally 中的 return 覆盖了 try 中待返回的 1，还可能吞掉异常。finally 应只做清理，不返回结果；下面用 append 模拟清理动作，修复后返回 1。", '''def original():
    try:
        return 1
    finally:
        return 2

events = []
def f():
    try:
        return 1
    finally:
        events.append('cleaned')

assert original() == 2
assert f() == 1
assert events == ['cleaned']
'''),
99: ("默认列表只在函数定义时创建一次。依次调用 f(1)、f(2)，第一次当时返回 [1]，第二次返回 [1, 2]；保存的两次结果引用同一列表，之后都显示 [1, 2]。改用 None 哨兵，每次省略参数时创建新列表。", '''def f(x, items=[]):
    items.append(x)
    return items

first = f(1)
assert first == [1]
second = f(2)
assert first is second and first == [1, 2]

def fixed(x, items=None):
    if items is None:
        items = []
    items.append(x)
    return items

assert fixed(1) == [1]
assert fixed(2) == [2]
explicit = []
assert fixed(3, explicit) is explicit
'''),
100: ("sorted 接受可迭代对象，返回新列表；list.sort 原地排序并返回 None。二者均稳定：排序键相等时保留原相对顺序，reverse=True 也保持这一性质。", '''values = [3, 1, 2]
ordered = sorted(values)
assert ordered == [1, 2, 3] and values == [3, 1, 2]
assert values.sort() is None
assert values == [1, 2, 3]
pairs = [(2, 'b'), (1, 'a'), (2, 'c')]
assert sorted(pairs, key=lambda pair: pair[0]) == [(1, 'a'), (2, 'b'), (2, 'c')]
assert sorted((3, 1, 2)) == [1, 2, 3]
'''),
}
