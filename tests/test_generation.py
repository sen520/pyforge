import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyforge.generation import APIHTTPError, check_answer, generate


def response(content="```python\ndef f():\n    return 1\n```", reason="stop"):
    return {"model": "test-teacher", "choices": [{"message": {"content": content, "reasoning_content": "not training data"}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


class GenerationTests(unittest.TestCase):
    def test_checks_reject_truncation_and_syntax(self):
        self.assertIsNotNone(check_answer(response())[0])
        self.assertIsNone(check_answer(response(reason="length"))[0])
        self.assertIsNone(check_answer(response("```python\ninvalid syntax!\n```"))[0])
        self.assertIsNone(check_answer(response("<think>reason</think>"))[0])

    def test_resume_and_split_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "questions.jsonl"
            rows = [{"id": "seed-00"+str(i), "group_id": str(i), "source": "pyforge_original_seed_v1", "prompt": "question "+str(i),
                "answer_requirements": "code", "task_type": "algorithm", "language": "zh", "license": "self_owned"} for i in range(3)]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            calls = []
            def api(key, payload, timeout):
                calls.append(payload)
                return response()
            with patch("pyforge.generation.credentials", return_value=("secret-for-test", "test-teacher")):
                summary = generate(source, root / "run", limit=1, api=api)
                self.assertEqual(summary["completed"], 1)
                summary = generate(source, root / "run", limit=3, api=api)
                self.assertEqual(summary["completed"], 3)
                self.assertEqual(len(calls), 3)
                generate(source, root / "run", limit=3, api=api)
                self.assertEqual(len(calls), 3)
            text = (root / "run/candidates.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("secret-for-test", text)
            self.assertNotIn("not training data", text)
            self.assertEqual({json.loads(line)["split"] for line in text.splitlines()}, {"train", "validation", "test"})

    def test_uncertain_request_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Path(__file__).resolve().parents[1] / "datasets/seed_questions.jsonl"
            def fail(*args):
                raise RuntimeError("timeout")
            with patch("pyforge.generation.credentials", return_value=("test-key", "test-teacher")):
                with self.assertRaises(RuntimeError):
                    generate(source, root, api=fail)
                with self.assertRaisesRegex(ValueError, "未确认"):
                    generate(source, root, api=fail)

    def test_old_http_error_cannot_hide_new_uncertain_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Path(__file__).resolve().parents[1] / "datasets/seed_questions.jsonl"
            def http_fail(*args):
                raise APIHTTPError(503)
            def uncertain_fail(*args):
                raise RuntimeError("timeout")
            with patch("pyforge.generation.credentials", return_value=("test-key", "test-teacher")):
                with self.assertRaises(APIHTTPError):
                    generate(source, root, api=http_fail)
                with self.assertRaises(RuntimeError):
                    generate(source, root, api=uncertain_fail)
                with self.assertRaisesRegex(ValueError, "未确认"):
                    generate(source, root, api=http_fail)
                calls = []
                def success(key, payload, timeout):
                    calls.append(payload)
                    return response()
                summary = generate(source, root, limit=2, api=success, workers=2, skip_uncertain=True)
                self.assertEqual(len(calls), 2)
                self.assertEqual(summary["uncertain_ids"], ["seed-001"])


if __name__ == "__main__":
    unittest.main()
