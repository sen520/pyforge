"""提取 doocs 题面，不执行或输出题解代码。"""

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "pre", "li", "br", "div"):
            self.parts.append("\n")
        if tag == "img":
            attrs = dict(attrs)
            self.parts.append("[图片：{} {}]".format(attrs.get("alt", ""), attrs.get("src", "")))
        if tag == "sup":
            self.parts.append("^(")
        if tag == "sub":
            self.parts.append("_(")

    def handle_endtag(self, tag):
        if tag in ("p", "pre", "li", "div"):
            self.parts.append("\n")
        if tag in ("sup", "sub"):
            self.parts.append(")")

    def handle_data(self, data):
        self.parts.append(data)


def extract_problem(text):
    title = re.search(r"^# \[(\d+)\. (.+?)\]\((https?://[^)]+)\)", text, re.M)
    description = re.search(r"<!-- description:start -->(.*?)<!-- description:end -->", text, re.S)
    difficulty = re.search(r"^difficulty:\s*(.+)$", text, re.M)
    if not title or not description:
        raise ValueError("缺少题号、标题或题面边界")
    signatures = []
    for code in re.findall(r"```python\s*\n(.*?)```", text, re.S):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not method.name.startswith("_") or method.name == "__init__"):
                        methods.append("    def {}({}): ...".format(method.name, ast.unparse(method.args)))
                if methods:
                    signatures.append("class {}:\n{}".format(node.name, "\n".join(methods)))
        if signatures:
            break
    if not signatures:
        raise ValueError("无可提取的 Python 类接口")
    parser = TextExtractor()
    parser.feed(description.group(1))
    plain = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(parser.parts)).strip()
    tags = re.search(r"^tags:\s*\n((?:[ \t]+-.*\n)+)", text, re.M)
    return {"problem_id": int(title.group(1)), "title": title.group(2), "problem_url": title.group(3),
            "difficulty": difficulty.group(1).strip() if difficulty else "未知",
            "tags": re.findall(r"-\s*(.+)", tags.group(1)) if tags else [],
            "description": plain,
            "interface": next((s for s in signatures if s.startswith("class Solution:")), signatures[0]),
            "contains_images": "<img" in description.group(1)}


def replace_interface(prompt, old, new):
    """只替换接口代码块，避免题面和答案逻辑进入本次修复。"""
    marker = "接口：\n```python\n" + old + "\n```"
    if prompt.count(marker) != 1:
        raise ValueError("题面必须有唯一接口块")
    return prompt.replace(marker, "接口：\n```python\n" + new + "\n```", 1)


def build_validation(source, destination, overlap_path):
    source, destination = Path(source).resolve(), Path(destination)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    overlap = json.loads(Path(overlap_path).read_text(encoding="utf-8"))
    excluded_ids = {int(key) for key in overlap["problem_ids"]}
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("目标目录非空，拒绝覆盖冻结数据")
    rows, excluded, skipped = [], [], []
    for path in sorted((source / "solution").glob("*/*/README.md")):
        try:
            raw = path.read_text(encoding="utf-8")
            problem = extract_problem(raw)
        except (ValueError, UnicodeError) as exc:
            skipped.append({"path": path.relative_to(source).as_posix(), "reason": str(exc)})
            continue
        problem_id = problem["problem_id"]
        if problem_id in excluded_ids:
            excluded.append({"problem_id": problem_id, "reason": overlap["problem_ids"][str(problem_id)]})
            continue
        relative = path.relative_to(source).as_posix()
        rows.append({"id": "doocs-leetcode-{:04d}".format(problem_id), "split": "validation",
            "group_id": "leetcode-{}".format(problem_id), "source": "doocs/leetcode", "source_revision": revision,
            "source_path": relative, "source_url": "https://github.com/doocs/leetcode/blob/{}/{}".format(revision, quote(relative)),
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(), "license": "CC-BY-SA-4.0",
            "attribution": "doocs/leetcode contributors; original problem: LeetCode",
            "changes": "仅提取中文题面；HTML 转文本；Python 实现替换为接口占位；不含题解",
            **problem, "prompt": "请用 Python 3.12 解答下面的题目。保留给定类与公开方法接口，输出完整代码。\n\n" + problem["title"] + "\n\n" + problem["description"] + "\n\n接口：\n```python\n" + problem["interface"] + "\n```",
            "status": "candidate_requires_review_and_tests", "execution_ready": False})
    if not rows:
        raise ValueError("未提取到可用题目，请确认检出了 solution/*/*/README.md")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "questions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (destination / "LICENSE").write_bytes((source / "LICENSE").read_bytes())
    manifest = {"source": "https://github.com/doocs/leetcode", "source_revision": revision,
        "license": "CC-BY-SA-4.0", "count": len(rows), "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "questions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "overlap_rules_sha256": hashlib.sha256(Path(overlap_path).read_bytes()).hexdigest(),
        "excluded_known_overlap": excluded, "skipped": skipped,
        "limitations": ["语义近似排除表并非完整去污染保证", "无隐藏测试，不可直接报告 pass@1", "公开题可能已被基础模型预训练使用"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
