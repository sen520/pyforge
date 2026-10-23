import unittest

from pyforge.leetcode import extract_problem


FIXTURE = '''---
difficulty: 简单
tags:
    - 数组
---
# [9999. 自造测试题](https://leetcode.cn/problems/test-fixture)
<!-- description:start -->
<p>返回 x&lt;10<sup>4</sup> 的标志。</p><pre>输入：x = 1\n输出：true</pre>
<!-- description:end -->
## 解法
秘密推理，不应进入题面。
```python
class Solution:
    def check(self, x: int) -> bool:
        return x < 10000  # 秘密实现
```
'''


class LeetcodeTests(unittest.TestCase):
    def test_extract_without_solution_leak(self):
        row = extract_problem(FIXTURE)
        self.assertEqual(row["problem_id"], 9999)
        self.assertEqual(row["tags"], ["数组"])
        self.assertIn("10^(4)", row["description"])
        self.assertIn("输入", row["description"])
        self.assertEqual(row["interface"], "class Solution:\n    def check(self, x: int): ...")
        for field in ("description", "interface"):
            self.assertNotIn("秘密", row[field])
            self.assertNotIn("return", row[field])

    def test_missing_markers_rejected(self):
        with self.assertRaises(ValueError):
            extract_problem(FIXTURE.replace("<!-- description:end -->", ""))

    def test_non_python_problem_rejected(self):
        with self.assertRaises(ValueError):
            extract_problem(FIXTURE.replace("```python", "```sql"))

    def test_image_dependency_marked(self):
        row = extract_problem(FIXTURE.replace("<p>返回", '<img src="test.png" alt="图"/><p>返回'))
        self.assertTrue(row["contains_images"])
        self.assertIn("test.png", row["description"])


if __name__ == "__main__":
    unittest.main()
