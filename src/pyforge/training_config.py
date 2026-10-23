"""训练参数、数据及恢复校验；不依赖 GPU 或训练框架。"""

import hashlib
import json
import math
from pathlib import Path

from .data import read_records


DEFAULTS = {"lora_dropout": 0.05, "save_steps": 100, "eval_steps": 100, "save_total_limit": 2,
    "warmup_ratio": 0.03, "weight_decay": 0.01, "max_grad_norm": 1.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def load_config(path):
    # Windows PowerShell 的 Set-Content -Encoding utf8 可能写入 BOM。
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"model", "revision", "max_length", "lora_rank", "lora_alpha", "gradient_accumulation_steps",
                "learning_rate", "epochs", "max_steps", "seed"}
    if not isinstance(raw, dict) or required - raw.keys():
        raise ValueError("训练配置缺少必填字段")
    unknown = raw.keys() - required - DEFAULTS.keys()
    if unknown:
        raise ValueError("未知训练配置：" + ", ".join(sorted(unknown)))
    config = {**DEFAULTS, **raw}
    for key in ("model", "revision"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(key + " 必须为非空字符串")
    for key in ("max_length", "lora_rank", "lora_alpha", "gradient_accumulation_steps", "save_steps", "eval_steps", "save_total_limit"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + " 必须为正整数")
    for key in ("learning_rate", "epochs", "max_grad_norm"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " 必须为有限正数")
    for key in ("lora_dropout", "warmup_ratio", "weight_decay"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(key + " 必须位于 [0, 1)")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    if type(config["max_steps"]) is not int or config["max_steps"] == 0 or config["max_steps"] < -1:
        raise ValueError("max_steps 必须为 -1 或正整数")
    if config["save_steps"] % config["eval_steps"]:
        raise ValueError("save_steps 必须为 eval_steps 的整数倍，确保最优检查点可恢复")
    targets = config["target_modules"]
    if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t for t in targets) or len(set(targets)) != len(targets):
        raise ValueError("target_modules 必须为非空且无重复的模块名列表")
    return config


def load_splits(directory):
    directory = Path(directory)
    rows = {s: read_records(directory / (s + ".jsonl")) for s in ("train", "validation", "test")}
    for field in ("id", "group_id"):
        values = [{row[field] for row in records} for records in rows.values()]
        if any(values[a] & values[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("跨集合 {} 泄漏".format(field))
    messages = [{json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in records} for records in rows.values()]
    if any(messages[a] & messages[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("跨集合重复对话泄漏")
    hashes = {s: hashlib.sha256((directory / (s + ".jsonl")).read_bytes()).hexdigest() for s in rows}
    return rows, hashes


def check_resume(checkpoint, output, config, data_hashes):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if checkpoint.parent != output:
        raise ValueError("恢复检查点必须位于本次输出目录中")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise ValueError("检查点不完整，缺少 " + name)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != config or manifest["data_sha256"] != data_hashes:
        raise ValueError("恢复时配置或数据哈希与原运行不一致")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] >= state["max_steps"]:
        raise ValueError("检查点已达到计划总步数；如需新实验请使用新的输出目录")
    return manifest, state
