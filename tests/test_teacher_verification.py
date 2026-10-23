import ast
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_teacher_data.py"))


class TeacherVerificationTests(unittest.TestCase):
    def test_all_harnesses_are_valid_python(self):
        self.assertEqual(set(MODULE["CHECKS"]), set(range(1,86)))
        for checks in MODULE["CHECKS"].values():
            ast.parse(MODULE["HELPER"].replace("{checks}",checks))

    def test_remove_embedded_tests_without_changing_function(self):
        code="import pytest\ndef f(x):\n return x+1\n@pytest.fixture\ndef fixture_value(): return 1\ndef test_f(): assert f(1)==2"
        result=MODULE["implementation_only"](code)
        self.assertNotIn("pytest",result)
        self.assertNotIn("test_f",result)
        self.assertNotIn("fixture_value",result)
        original=ast.parse(code).body[1]
        cleaned=ast.parse(result).body[0]
        self.assertEqual(ast.dump(original),ast.dump(cleaned))
        # 反斜杠用例是真实路径分隔符，不能被错误转义为退格。
        tree=ast.parse(MODULE["HELPER"].replace("{checks}",MODULE["CHECKS"][55]))
        self.assertIn('a\\b',[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)])

    def test_unverified_not_exported_and_extra_blocks_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"candidates.jsonl"
            response="说明\n```python\ndef two_sum(a,b): return None\n```\n```python\nimport nonexistent_test_module\n```"
            rows=[{"id":identity,"split":"train","messages":[{"role":"assistant","content":response}],
                "verification":{"syntax":True,"execution":False,"human_review":False}} for identity in ("seed-001","seed-086")]
            source.write_text("\n".join(json.dumps(row) for row in rows),encoding="utf-8")
            globals_=MODULE["verify"].__globals__
            with patch.dict(globals_,docker_environment=lambda image:"sha256:test",run_isolated=lambda *a,**k:{"status":"passed"}):
                result=MODULE["verify"](source,root/"export")
                self.assertEqual(result["verified"],1)
                self.assertEqual(result["pending"],1)
                text=(root/"export/train.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("nonexistent_test_module",text)
                self.assertNotIn("seed-086",text)
                again=MODULE["verify"](source,root/"export")
                self.assertEqual(again,result)


if __name__ == "__main__":
    unittest.main()
