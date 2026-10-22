"""不依赖深度学习框架的数据校验与分组划分。"""

import hashlib
import json
import random
from pathlib import Path


def read_records(path):
    records, ids, fingerprints = [], set(), {}
    required = ("id", "task_type", "language", "source", "license", "group_id")
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("记录必须是对象")
                for key in required:
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError("缺少非空字符串字段：" + key)
                if record["id"] in ids:
                    raise ValueError("重复 id：" + record["id"])
                messages = record.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages 必须是非空数组")
                conversation = messages[1:] if messages[0].get("role") == "system" else messages
                if len(conversation) < 2 or len(conversation) % 2:
                    raise ValueError("对话必须包含完整 user/assistant 回合")
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise ValueError("message 必须是对象")
                    if not isinstance(message.get("content"), str) or not message["content"].strip():
                        raise ValueError("消息内容不能为空")
                for i, message in enumerate(conversation):
                    if message.get("role") != ("user" if i % 2 == 0 else "assistant"):
                        raise ValueError("角色顺序必须为 user/assistant 交替")
                fingerprint = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if fingerprint in fingerprints:
                    raise ValueError("重复对话，与 {} 重复".format(fingerprints[fingerprint]))
                fingerprints[fingerprint] = record["id"]
                ids.add(record["id"])
                records.append(record)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError("{}:{}: {}".format(path, line_number, exc)) from exc
    if not records:
        raise ValueError("数据集不能为空")
    return records


def split_records(records, seed=42):
    groups = sorted({record["group_id"] for record in records})
    if len(groups) < 3:
        raise ValueError("至少需要 3 个独立 group_id，才能划分三个非空集合")
    random.Random(seed).shuffle(groups)
    holdout = max(1, int(len(groups) * 0.05))
    assignments = {}
    for i, group in enumerate(groups):
        assignments[group] = "test" if i < holdout else "validation" if i < 2 * holdout else "train"
    result = {key: [] for key in ("train", "validation", "test")}
    for record in records:
        result[assignments[record["group_id"]]].append(record)
    return result


def prepare(source, destination, seed=42):
    records = read_records(source)
    result = split_records(records, seed)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("输出目录非空，请使用新目录以避免覆盖数据")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "splits": {}}
    for name, rows in result.items():
        path = destination / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        manifest["splits"][name] = {"count": len(rows), "ids": [r["id"] for r in rows], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / "数据清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
