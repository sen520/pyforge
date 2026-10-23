"""Codex 直接编写的种子题实现；不读取任何服务端答案。"""


def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return seen[target - value], i
        seen[value] = i
    return None


def unique_sorted(nums):
    return sorted(set(nums))


def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def binary_search(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(nums) and nums[left] == target else -1


def lower_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def rotate_right(nums, k):
    if not nums:
        return []
    k %= len(nums)
    return nums[-k:] + nums[:-k] if k else list(nums)


def move_zeros(nums):
    values = [x for x in nums if x != 0]
    return values + [0] * (len(nums) - len(values))


def max_subarray(nums):
    if not nums:
        raise ValueError("nums must not be empty")
    current = best = nums[0]
    for index in range(1, len(nums)):
        value = nums[index]
        current = max(value, current + value)
        best = max(best, current)
    return best


def longest_increasing_run(nums):
    best = current = 0
    for i, value in enumerate(nums):
        current = current + 1 if i and value > nums[i - 1] else 1
        best = max(best, current)
    return best


def product_except_self(nums):
    result = [1] * len(nums)
    prefix = 1
    for i, value in enumerate(nums):
        result[i] = prefix
        prefix *= value
    suffix = 1
    for i in range(len(nums) - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intersection(a, b):
    return sorted(set(a) & set(b))


def first_unique(text):
    from collections import Counter
    counts = Counter(text)
    return next((i for i, ch in enumerate(text) if counts[ch] == 1), -1)


def is_palindrome(text):
    filtered = [c.lower() for c in text if c.isascii() and c.isalnum()]
    return filtered == filtered[::-1]


def anagram_groups(words):
    groups = {}
    for word in words:
        counts = [0] * 26
        for ch in word:
            counts[ord(ch) - ord('a')] += 1
        groups.setdefault(tuple(counts), []).append(word)
    return list(groups.values())


def longest_unique_substring(text):
    seen = {}
    left = best = 0
    for right, ch in enumerate(text):
        left = max(left, seen.get(ch, -1) + 1)
        seen[ch] = right
        best = max(best, right - left + 1)
    return best


def valid_brackets(text):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    if any(ch not in '()[]{}' for ch in text):
        raise ValueError("unexpected character")
    for ch in text:
        if ch in '([{':
            stack.append(ch)
        elif not stack or stack.pop() != pairs[ch]:
            return False
    return not stack


def run_length_encode(text):
    result = []
    for ch in text:
        if result and result[-1][0] == ch:
            result[-1][1] += 1
        else:
            result.append([ch, 1])
    return result


def substring_index(text, pattern):
    if not pattern:
        return 0
    prefix = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]:
            j = prefix[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        prefix[i] = j
    j = 0
    for i, ch in enumerate(text):
        while j and ch != pattern[j]:
            j = prefix[j - 1]
        if ch == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - len(pattern) + 1
    return -1


def common_prefix(words):
    if not words:
        return ''
    result = []
    for column in zip(*words):
        if any(ch != column[0] for ch in column):
            break
        result.append(column[0])
    return ''.join(result)


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def is_prime(n):
    from math import isqrt
    if n < 2:
        return False
    return all(n % d for d in range(2, isqrt(n) + 1))


def sieve(n):
    from math import isqrt
    if n < 2:
        return []
    prime = [True] * (n + 1)
    prime[0] = prime[1] = False
    for p in range(2, isqrt(n) + 1):
        if prime[p]:
            for multiple in range(p * p, n + 1, p):
                prime[multiple] = False
    return [i for i in range(n + 1) if prime[i]]


def fast_power(base, exponent):
    result = 1
    while exponent:
        if exponent & 1:
            result *= base
        base *= base
        exponent //= 2
    return result


def popcount(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def climb_stairs(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def min_coins(coins, amount):
    dp = [0] + [amount + 1] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                dp[value] = min(dp[value], dp[value - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1


def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for limit in range(capacity, weight - 1, -1):
            dp[limit] = max(dp[limit], dp[limit - weight] + value)
    return dp[capacity]


def longest_common_subsequence(a, b):
    dp = [0] * (len(b) + 1)
    for ch in a:
        diagonal = 0
        for j, other in enumerate(b, 1):
            old = dp[j]
            dp[j] = diagonal + 1 if ch == other else max(dp[j], dp[j - 1])
            diagonal = old
    return dp[-1]


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ch in enumerate(a, 1):
        previous = dp[0]
        dp[0] = i
        for j, other in enumerate(b, 1):
            old = dp[j]
            dp[j] = previous if ch == other else 1 + min(previous, dp[j], dp[j - 1])
            previous = old
    return dp[-1]


def grid_paths(rows, cols):
    if not rows or not cols:
        return 0
    dp = [1] * cols
    for _ in range(1, rows):
        for j in range(1, cols):
            dp[j] += dp[j - 1]
    return dp[-1]


def shortest_path(graph, start, goal):
    from collections import deque
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        for neighbor in graph.get(node, []):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return None


def connected_components(n, edges):
    graph = [[] for _ in range(n)]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    visited = set()
    count = 0
    for root in range(n):
        if root in visited:
            continue
        count += 1
        visited.add(root)
        stack = [root]
        while stack:
            for neighbor in graph[stack.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return count


def topological_order(n, edges):
    from collections import deque
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    for a, b in edges:
        graph[a].append(b)
        indegree[b] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return order if len(order) == n else None


def flood_fill(image, row, col, color):
    result = [line[:] for line in image]
    original = result[row][col]
    if color == original:
        return result
    result[row][col] = color
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < len(result) and 0 <= nc < len(result[0]) and result[nr][nc] == original:
                result[nr][nc] = color
                stack.append((nr, nc))
    return result


def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    top, bottom, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
    result = []
    while top <= bottom and left <= right:
        result.extend(matrix[top][left:right+1])
        top += 1
        for row in range(top, bottom+1):
            result.append(matrix[row][right])
        right -= 1
        if top <= bottom:
            result.extend(matrix[bottom][left:right+1][::-1])
            bottom -= 1
        if left <= right:
            for row in range(bottom, top-1, -1):
                result.append(matrix[row][left])
            left += 1
    return result


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def k_largest(nums, k):
    import heapq
    if k == 0:
        return []
    heap = []
    for value in nums:
        if len(heap) < k:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)


def inversion_count(nums):
    def count(values):
        if len(values) < 2:
            return list(values), 0
        mid = len(values) // 2
        left, a = count(values[:mid])
        right, b = count(values[mid:])
        merged = []
        i = j = 0
        total = a + b
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                total += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, total
    return count(nums)[1]


def parse_ints(text):
    return [] if not text.strip() else [int(field.strip()) for field in text.split(',')]


def chunked(items, size):
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    return [items[i:i+size] for i in range(0, len(items), size)]


def flatten_once(items):
    return [value for item in items for value in (item if isinstance(item, list) else [item])]


def dedupe_records(records, key):
    seen = set()
    result = []
    for record in records:
        value = record[key]
        if value not in seen:
            seen.add(value)
            result.append(record)
    return result


def sort_records(records):
    return sorted(records, key=lambda record: (record['age'], record['name']))


def word_counts(text):
    import re
    from collections import Counter
    return dict(Counter(word.lower() for word in re.findall(r'[A-Za-z]+', text)))


def parse_query(query):
    from urllib.parse import parse_qs
    return parse_qs(query, keep_blank_values=True)


def build_url(base, params):
    from urllib.parse import urlencode
    query = urlencode(params, doseq=True)
    return base + ('?' + query if query else '')


def read_json(path):
    import json
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv_rows(path):
    import csv
    with open(path, encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path, rows, fields):
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def iter_nonempty_lines(path):
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            value = line.rstrip('\r\n')
            if value.strip():
                yield value


def count_extensions(paths):
    from collections import Counter
    from pathlib import Path
    return dict(Counter(Path(path).suffix.lower() or '<none>' for path in paths))


def safe_child(root, name):
    from pathlib import Path, PureWindowsPath
    if not name or name in ('.', '..') or '/' in name or '\\' in name or PureWindowsPath(name).drive:
        raise ValueError("name must be a single relative filename")
    return Path(root) / name


def parse_iso_date(text):
    import re
    from datetime import date
    if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', text):
        raise ValueError("expected YYYY-MM-DD")
    return date.fromisoformat(text)


def date_range(start, end):
    from datetime import timedelta
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def group_by_day(events):
    from datetime import datetime, timezone
    result = {}
    for event in events:
        timestamp = datetime.fromisoformat(event['timestamp'])
        if timestamp.utcoffset() is None:
            raise ValueError("timestamp must contain a timezone")
        day = timestamp.astimezone(timezone.utc).date()
        result.setdefault(day, []).append(event)
    return result


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def retry_call(func, attempts):
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    for attempt in range(attempts):
        try:
            return func()
        except TimeoutError:
            if attempt == attempts - 1:
                raise


def memoize(func):
    from functools import wraps
    cache = {}
    @wraps(func)
    def wrapped(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapped


def timed_call(func, *args, **kwargs):
    from time import perf_counter
    start = perf_counter()
    result = func(*args, **kwargs)
    return result, perf_counter() - start


def redact_secrets(record):
    return {key: '***' if key.lower() in {'password', 'token', 'api_key'} else value for key, value in record.items()}


def validate_config(config):
    host, port = config.get('host'), config.get('port')
    if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid host or port")
    return {'host': host, 'port': port}


def merge_defaults(defaults, overrides):
    return {**defaults, **overrides}


def append_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items


def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def remove_even(nums):
    return [value for value in nums if value % 2]


def make_grid(rows, cols):
    return [[0] * cols for _ in range(rows)]


def multipliers():
    return [lambda x, factor=i: factor * x for i in range(3)]


def parse_number(text):
    try:
        return int(text)
    except ValueError:
        return None


def sorted_copy(nums):
    return sorted(nums)


def total_and_count(values):
    total = count = 0
    for value in values:
        total += value
        count += 1
    return total, count


def counts(items):
    from collections import Counter
    return dict(Counter(items))


def maximum(nums):
    if not nums:
        raise ValueError("nums must not be empty")
    return max(nums)


def is_quit(command):
    return command == 'quit'


def get_value(data, key, default):
    return data.get(key, default)


def clone_records(records):
    from copy import deepcopy
    return deepcopy(records)


def factorial(n):
    if n < 0:
        raise ValueError("n must be nonnegative")
    result = 1
    for factor in range(2, n + 1):
        result *= factor
    return result


def add_money(a, b):
    from decimal import Decimal, localcontext
    x, y = Decimal(a), Decimal(b)
    # 覆盖输入有效数字及小数位，避免默认 28 位精度截断合法长金额。
    precision = max(len(x.as_tuple().digits), len(y.as_tuple().digits))
    precision += abs(x.as_tuple().exponent - y.as_tuple().exponent) + 2
    with localcontext() as context:
        context.prec = max(28, precision)
        return x + y


def remove_txt(name):
    return name.removesuffix('.txt')


def common(a, b):
    return sorted(set(a) & set(b))


def clean(data):
    return {key: value for key, value in data.items() if value is not None}


def inclusive_sum(start, end):
    if start > end:
        return 0
    return (start + end) * (end - start + 1) // 2


def valid_port(port):
    return type(port) is int and 1 <= port <= 65535
