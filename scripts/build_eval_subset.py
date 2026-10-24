"""从验证候选集中冻结 20 道纯函数题和独立编写的测试。"""

import argparse
import hashlib
import json
from pathlib import Path


# 每项为题号、方法、比较规则、(位置参数列表, 期望结果)；不使用上游参考实现。
CASES = [
    (9, "isPalindrome", "exact", [([0], True), ([7], True), ([121], True), ([-121], False), ([10], False), ([1001], True)]),
    (11, "maxArea", "exact", [([[1, 1]], 1), ([[1, 2, 1]], 2), ([[1, 8, 6, 2, 5, 4, 8, 3, 7]], 49), ([[4, 3, 2, 1, 4]], 16), ([[0, 0]], 0)]),
    (15, "threeSum", "unordered_nested", [([[-1, 0, 1, 2, -1, -4]], [[-1, -1, 2], [-1, 0, 1]]), ([[0, 0, 0, 0]], [[0, 0, 0]]), ([[1, 2, 3]], []), ([[-2, 0, 1, 1, 2]], [[-2, 0, 2], [-2, 1, 1]]), ([[-1, 0, 1]], [[-1, 0, 1]])]),
    (17, "letterCombinations", "unordered", [(["2"], list("abc")), (["7"], list("pqrs")), (["23"], [a+b for a in "abc" for b in "def"]), (["79"], [a+b for a in "pqrs" for b in "wxyz"]), (["9"], list("wxyz"))]),
    (22, "generateParenthesis", "unordered", [([1], ["()"]), ([2], ["(())", "()()"]), ([3], ["((()))", "(()())", "(())()", "()(())", "()()()"])]),
    (58, "lengthOfLastWord", "exact", [(["a"], 1), (["hello world"], 5), (["  fly me   to   the moon  "], 4), (["word   "], 4), (["a b"], 1)]),
    (66, "plusOne", "exact", [([[0]], [1]), ([[9]], [1, 0]), ([[1, 2, 9]], [1, 3, 0]), ([[9, 9]], [1, 0, 0]), ([[1, 0, 0]], [1, 0, 1])]),
    (67, "addBinary", "exact", [(["0", "0"], "0"), (["1", "1"], "10"), (["1010", "1011"], "10101"), (["111", "1"], "1000"), (["1", "0"], "1")]),
    (69, "mySqrt", "exact", [([0], 0), ([1], 1), ([8], 2), ([16], 4), ([2147395600], 46340), ([2147483647], 46340)]),
    (121, "maxProfit", "exact", [([[7, 1, 5, 3, 6, 4]], 5), ([[7, 6, 4, 3, 1]], 0), ([[1]], 0), ([[2, 2]], 0), ([[2, 4, 1]], 2)]),
    (136, "singleNumber", "exact", [([[1]], 1), ([[2, 2, 1]], 1), ([[4, 1, 2, 1, 2]], 4), ([[-1, -1, -2]], -2), ([[0, 1, 1]], 0)]),
    (169, "majorityElement", "exact", [([[1]], 1), ([[3, 2, 3]], 3), ([[2, 2, 1, 1, 1, 2, 2]], 2), ([[-1, -1, 0]], -1), ([[0, 0]], 0)]),
    (202, "isHappy", "exact", [([1], True), ([19], True), ([2], False), ([7], True), ([4], False)]),
    (205, "isIsomorphic", "exact", [(["egg", "add"], True), (["foo", "bar"], False), (["paper", "title"], True), (["ab", "aa"], False), (["a", "b"], True)]),
    (217, "containsDuplicate", "exact", [([[1]], False), ([[1, 2, 3]], False), ([[1, 2, 1]], True), ([[0, 0]], True), ([[-1, 0, 1, -1]], True)]),
    (219, "containsNearbyDuplicate", "exact", [([[1, 2, 3, 1], 3], True), ([[1, 2, 3, 1], 2], False), ([[1, 1], 0], False), ([[1], 1], False), ([[1, 0, 1, 1], 1], True)]),
    (228, "summaryRanges", "exact", [([[]], []), ([[0]], ["0"]), ([[0, 1, 2, 4, 5, 7]], ["0->2", "4->5", "7"]), ([[-3, -2, -1, 1]], ["-3->-1", "1"]), ([[-2147483648, 2147483647]], ["-2147483648", "2147483647"])]),
    (268, "missingNumber", "exact", [([[0]], 1), ([[1]], 0), ([[3, 0, 1]], 2), ([[0, 1]], 2), ([[2, 1]], 0)]),
    (383, "canConstruct", "exact", [(["a", "b"], False), (["aa", "ab"], False), (["aa", "aab"], True), (["abc", "cbad"], True), (["zzz", "zz"], False)]),
]


def build(source, output):
    source, output = Path(source), Path(output)
    rows = {r["problem_id"]: r for r in map(json.loads, source.read_text(encoding="utf-8").splitlines())}
    selected = []
    for number, method, comparator, cases in CASES:
        row = dict(rows[number])
        row.update(method=method, comparator=comparator, tests=[{"args": args, "expected": expected} for args, expected in cases],
                   execution_ready=True, status="executable_subset_v1", tests_source="pyforge independently authored; not official hidden tests")
        selected.append(row)
    # Sudoku：从合法基础盘面构造行/列/宫冲突，期望结果独立指定。
    base = [list(s) for s in ("53..7....", "6..195...", ".98....6.", "8...6...3", "4..8.3..1", "7...2...6", ".6....28.", "...419..5", "....8..79")]
    boards = [(base, True), ([["."] * 9 for _ in range(9)], True)]
    for r, c, value in ((0, 1, "5"), (2, 0, "5"), (1, 1, "3")):
        board = [line[:] for line in base]
        board[r][c] = value
        boards.append((board, False))
    row = dict(rows[36])
    row.update(method="isValidSudoku", comparator="exact", tests=[{"args": [board], "expected": expected} for board, expected in boards],
               execution_ready=True, status="executable_subset_v1", tests_source="pyforge independently authored; not official hidden tests")
    selected.append(row)
    selected.sort(key=lambda r: r["problem_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": 1, "count": len(selected), "test_count": sum(len(r["tests"]) for r in selected),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "ids": [r["id"] for r in selected], "note": "可执行小样本验证子集，不是完整隐藏测试或最终测试集"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="datasets/leetcode/benchmarks/doocs-leetcode/questions.jsonl")
    parser.add_argument("--output", default="datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))
