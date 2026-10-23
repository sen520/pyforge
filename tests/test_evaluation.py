import json
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from pyforge.evaluation import compare, evaluate, extract_code, run_isolated, docker_environment


class EvaluationTests(unittest.TestCase):
    def test_extract_does_not_execute(self):
        self.assertEqual(extract_code("```python\nraise Exception('not executed')\n```"), "raise Exception('not executed')\n")
        for text in ("", "```python\nx = 1", "```python\nx=1\n```\n```python\nx=2\n```"):
            with self.assertRaises(ValueError):
                extract_code(text)
        with self.assertRaises(SyntaxError):
            extract_code("this is invalid python!")

    def test_comparison_preserves_multiplicity_and_types(self):
        self.assertTrue(compare(["b", "a"], ["a", "b"], "unordered"))
        self.assertFalse(compare(["a", "a"], ["a"], "unordered"))
        self.assertTrue(compare([[1, -1, 0]], [[-1, 0, 1]], "unordered_nested"))
        self.assertFalse(compare([[0, 0, 0], [0, 0, 0]], [[0, 0, 0]], "unordered_nested"))
        self.assertFalse(compare(1, True, "exact"))
        self.assertFalse(compare([True], [1], "exact"))

    def test_missing_answers_count_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, predictions, report = (root / name for name in ("suite.jsonl", "predictions.jsonl", "report.json"))
            rows = [{"id": str(i), "execution_ready": True, "tests": [{"args": [], "expected": 1}], "method": "f", "comparator": "exact"} for i in range(3)]
            suite.write_text("\n".join(json.dumps(row) for row in rows))
            predictions.write_text(json.dumps({"id": "0", "response": "class Solution: pass"}) + "\n" + json.dumps({"id": "1", "response": "bad code!"}))
            with patch("pyforge.evaluation.docker_environment", return_value="sha256:test"), patch("pyforge.evaluation.run_isolated", return_value={"status": "passed"}):
                result = evaluate(suite, predictions, report)
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["results"][2]["status"], "missing_prediction")
            with self.assertRaises(ValueError):
                evaluate(suite, predictions, report)

    @unittest.skipUnless(os.environ.get("PYFORGE_DOCKER_TESTS") == "1", "需要 Docker 集成测试环境")
    def test_real_docker_worker(self):
        image = docker_environment("python:3.12-slim")
        problem = {"method": "f", "comparator": "exact", "tests": [{"args": [2], "expected": 3}]}
        self.assertEqual(run_isolated("class Solution:\n def f(self, x): return x+1", problem, image)["status"], "passed")
        self.assertEqual(run_isolated("class Solution:\n def f(self, x): return x", problem, image)["status"], "wrong_answer")
        self.assertEqual(run_isolated("raise ValueError('test')", problem, image)["status"], "runtime_error")
        self.assertEqual(run_isolated("while True: pass", problem, image, timeout=2)["status"], "timeout")

    @unittest.skipUnless(os.environ.get("PYFORGE_DOCKER_TESTS") == "1", "需要 Docker 集成测试环境")
    def test_full_suite_in_docker(self):
        root = Path(__file__).resolve().parents[1]
        suite = root / "datasets/benchmarks/doocs-leetcode/executable_v1.jsonl"
        source = (Path(__file__).parent / "reference_solutions.py").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in suite.read_text(encoding="utf-8").splitlines()]
        with tempfile.TemporaryDirectory() as tmp:
            predictions, report = Path(tmp) / "predictions.jsonl", Path(tmp) / "report.json"
            predictions.write_text("\n".join(json.dumps({"id": row["id"], "response": source}) for row in rows), encoding="utf-8")
            result = evaluate(suite, predictions, report)
            self.assertEqual(result["passed"], 20, result["results"])

    @unittest.skipUnless(os.environ.get("PYFORGE_DOCKER_TESTS") == "1", "需要 Docker 集成测试环境")
    def test_container_restrictions(self):
        code = """import os, socket
class Solution:
 def f(self):
  if os.getuid() != 65534: return False
  try:
   open('/forbidden', 'w').close()
   return False
  except OSError: pass
  try:
   socket.create_connection(('1.1.1.1', 443), timeout=0.2).close()
   return False
  except OSError: return True
"""
        problem = {"method": "f", "comparator": "exact", "tests": [{"args": [], "expected": True}]}
        result = run_isolated(code, problem, docker_environment("python:3.12-slim"))
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
