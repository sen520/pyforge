import json
import tempfile
import unittest
from pathlib import Path

from pyforge.data import prepare, read_records, split_records
from pyforge.encoding import encode_record


def record(i=0, group=None):
    return {"id": str(i), "task_type": "code_generation", "language": "zh", "source": "test",
            "license": "self_owned", "group_id": group or str(i),
            "messages": [{"role": "user", "content": "question " + str(i)}, {"role": "assistant", "content": "answer"}]}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        tokens = []
        for message in messages:
            tokens += [1 if message["role"] == "user" else 2]
            tokens += [ord(char) for char in message["content"]] + [3]
        return tokens + ([2] if add_generation_prompt else [])


class DataTests(unittest.TestCase):
    def write(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_invalid_and_duplicate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            for rows in ([record(), record()], [{**record(), "messages": []}],
                         [{**record(), "messages": [None]}],
                         [{**record(), "messages": [{"role": "assistant", "content": "x"}]}],
                         [record(), {**record(1), "messages": record()["messages"]}]):
                self.write(path, rows)
                with self.assertRaises(ValueError):
                    read_records(path)

    def test_group_split_reproducible_and_disjoint(self):
        rows = [record(i, str(i // 2)) for i in range(40)]
        splits = split_records(rows)
        self.assertEqual(splits, split_records(rows))
        groups = [{row["group_id"] for row in values} for values in splits.values()]
        self.assertTrue(all(groups))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, splits.values())), 40)
        with self.assertRaises(ValueError):
            split_records([record()])

    def test_prepare_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.jsonl", Path(tmp) / "prepared"
            self.write(source, [record(i) for i in range(10)])
            manifest = prepare(source, output)
            self.assertEqual(sum(v["count"] for v in manifest["splits"].values()), 10)
            for name in manifest["splits"]:
                read_records(output / (name + ".jsonl"))
            with self.assertRaises(ValueError):
                prepare(source, output)

    def test_assistant_mask_and_long_filter(self):
        tokenizer = FakeTokenizer()
        row = record()
        row["messages"] += [{"role": "user", "content": "again"}, {"role": "assistant", "content": "ok"}]
        result = encode_record(row, tokenizer, 1000)
        expected = [ord(char) for char in "answer"] + [3] + [ord(char) for char in "ok"] + [3]
        self.assertEqual([label for label in result["labels"] if label != -100], expected)
        self.assertEqual(len(result["labels"]), len(result["input_ids"]))
        self.assertIsNone(encode_record(row, tokenizer, 2))

    def test_incompatible_template_is_rejected(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [999] if add_generation_prompt else super().apply_chat_template(messages, tokenize, False)
        with self.assertRaises(ValueError):
            encode_record(record(), Broken(), 1000)


if __name__ == "__main__":
    unittest.main()
