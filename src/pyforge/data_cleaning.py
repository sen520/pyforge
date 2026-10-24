"""数据清洗配置的严格校验与解释执行；不生成训练需求或答案。"""

from functools import cmp_to_key
import json
import math


SYSTEM_PROMPT = '''将中文数据清洗需求转换为配置，只输出一个 JSON 对象，不要解释或 Markdown。
格式为 {"steps":[...]}，按需求指定的顺序执行，不能添加未要求的操作。
支持四种操作，各操作只允许以下字段：
deduplicate: {"op":"deduplicate","columns":["列名"],"keep":"first"}，keep 为 first 或 last，按指定列组合去重，保留行保持原相对顺序。
drop_null: {"op":"drop_null","columns":["列名"],"how":"any"}，any 删除任意指定列为 null 的行，all 只删除全部指定列为 null 的行。空字符串、0、false 不是 null。
filter: {"op":"filter","column":"列名","operator":"ge","value":18}，保留满足条件的行。operator 支持 eq、ne、gt、ge、lt、le、in、not_in。in/not_in 使用非空值数组，其余使用非 null 标量。比较区分类型，数字与布尔值不同；null 在任何筛选中都被删除。严格区分大于与大于等于、小于与小于等于。
sort: {"op":"sort","by":[{"column":"列名","ascending":true}]}，by 按优先级排列，ascending 为布尔值，各列可独立升降序，稳定排序，null 始终最后。
列名与值使用需求指定的原值，保留数值、字符串、布尔值的类型。'''


def _scalar(value):
    return (type(value) in (str, bool, int) or
            type(value) is float and math.isfinite(value))


def _columns(value):
    return (isinstance(value, list) and bool(value) and
            all(isinstance(v, str) and v.strip() for v in value) and
            len(set(value)) == len(value))


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {'steps'}:
        raise ValueError('配置必须且只能包含 steps')
    if not isinstance(config['steps'], list) or not config['steps']:
        raise ValueError('steps 必须为非空数组')
    for step in config['steps']:
        if not isinstance(step, dict):
            raise ValueError('每一步必须为对象')
        op = step.get('op')
        if op == 'deduplicate':
            valid = (set(step) == {'op', 'columns', 'keep'} and
                     _columns(step['columns']) and step['keep'] in ('first', 'last'))
        elif op == 'drop_null':
            valid = (set(step) == {'op', 'columns', 'how'} and
                     _columns(step['columns']) and step['how'] in ('any', 'all'))
        elif op == 'filter':
            valid = (set(step) == {'op', 'column', 'operator', 'value'} and
                     isinstance(step['column'], str) and bool(step['column'].strip()) and
                     step['operator'] in ('eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not_in'))
            if valid:
                value = step['value']
                if step['operator'] in ('in', 'not_in'):
                    valid = isinstance(value, list) and bool(value) and all(_scalar(v) for v in value)
                else:
                    valid = _scalar(value)
                if valid and step['operator'] in ('gt', 'ge', 'lt', 'le'):
                    valid = type(value) in (str, int, float)
        elif op == 'sort':
            valid = set(step) == {'op', 'by'} and isinstance(step['by'], list) and bool(step['by'])
            if valid:
                valid = all(isinstance(v, dict) and set(v) == {'column', 'ascending'} and
                            isinstance(v['column'], str) and v['column'].strip() and
                            type(v['ascending']) is bool for v in step['by'])
                valid = valid and len({v['column'] for v in step['by']}) == len(step['by'])
        else:
            valid = False
        if not valid:
            raise ValueError('不合法的清洗操作：' + repr(step))
    return config


def parse_config(response):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('JSON 对象包含重复键：' + key)
            result[key] = value
        return result

    return validate_config(json.loads(response, object_pairs_hook=unique_object))


def _equal(a, b):
    # 整数与浮点数可数值比较，但 bool 不隐式等于 0/1。
    numbers = type(a) in (int, float) and type(b) in (int, float)
    return (numbers or type(a) is type(b)) and a == b


def _match(value, step):
    if value is None:
        return False
    other, op = step['value'], step['operator']
    if op in ('in', 'not_in'):
        found = any(_equal(value, item) for item in other)
        return found if op == 'in' else not found
    if op in ('eq', 'ne'):
        found = _equal(value, other)
        return found if op == 'eq' else not found
    compatible = (type(value) in (int, float) and type(other) in (int, float) or
                  type(value) is str and type(other) is str)
    if not compatible:
        raise ValueError('有序比较要求同类型字符串或数值')
    if op == 'gt':
        return value > other
    if op == 'ge':
        return value >= other
    if op == 'lt':
        return value < other
    return value <= other


def execute_config(config, rows):
    """只解释四种固定操作；从不执行样本或预测中的 Python 代码。"""
    validate_config(config)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('输入必须为行对象数组')
    result = [dict(row) for row in rows]
    for step in config['steps']:
        columns = (step['columns'] if 'columns' in step else
                   [v['column'] for v in step['by']] if 'by' in step else [step['column']])
        if any(column not in row for row in result for column in columns):
            raise ValueError('输入行缺少配置引用的列')
        if step['op'] == 'drop_null':
            predicate = any if step['how'] == 'any' else all
            result = [r for r in result if not predicate(r[c] is None for c in columns)]
        elif step['op'] == 'filter':
            result = [r for r in result if _match(r[step['column']], step)]
        elif step['op'] == 'deduplicate':
            keys, kept = [], []
            candidates = result if step['keep'] == 'first' else reversed(result)
            for row in candidates:
                values = [row[c] for c in columns]
                if not all(v is None or _scalar(v) for v in values):
                    raise ValueError('去重列只支持 JSON 标量与 null')
                if not any(all(_equal(a, b) for a, b in zip(values, key)) for key in keys):
                    keys.append(values)
                    kept.append(row)
            result = kept if step['keep'] == 'first' else list(reversed(kept))
        else:
            def compare(a, b):
                for order in step['by']:
                    left, right = a[order['column']], b[order['column']]
                    if left is None or right is None:
                        comparison = (left is None) - (right is None)
                        if comparison:
                            return comparison
                        continue
                    numeric = type(left) in (int, float) and type(right) in (int, float)
                    if not (numeric or type(left) is type(right) and type(left) in (str, bool)):
                        raise ValueError('排序列非空值类型必须一致')
                    comparison = (left > right) - (left < right)
                    if comparison:
                        return comparison if order['ascending'] else -comparison
                return 0
            result = sorted(result, key=cmp_to_key(compare))
    return result
