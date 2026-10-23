"""硅基流动教师数据生成；密钥只在内存使用，原始结果默认不提交。"""

import hashlib
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from .data import split_records
from .paths import PROJECT_ROOT


JOURNAL_LOCK = threading.Lock()


SYSTEM = "你是严谨的 Python 编程助手。使用 Python 3.12，准确遵守接口和约束。输出简短说明及一个完整 Python 代码块，不输出冗长思维过程。测试题使用 pytest，解释题提供可运行的演示或修复代码。"


class APIHTTPError(RuntimeError):
    def __init__(self, status, detail=""):
        self.status = status
        super().__init__("硅基流动 HTTP {}，本次不会自动重试{}".format(status, "："+detail if detail else ""))


def credentials():
    values = {}
    path = PROJECT_ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if separator and key.strip() in ("SILICONFLOW_API_KEY", "SILICONFLOW_MODEL"):
                values[key.strip()] = value.strip().strip("\"'")
    key = os.environ.get("SILICONFLOW_API_KEY") or values.get("SILICONFLOW_API_KEY")
    model = os.environ.get("SILICONFLOW_MODEL") or values.get("SILICONFLOW_MODEL") or "deepseek-ai/DeepSeek-V4-Pro"
    if not key:
        raise ValueError("请在项目根目录 .env 中配置 SILICONFLOW_API_KEY")
    return key, model


def call_api(api_key, payload, timeout=120):
    req = request.Request("https://api.siliconflow.cn/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise RuntimeError("响应过大；请求可能已计费，请检查 journal 后再处理")
            return json.loads(raw)
    except error.HTTPError as exc:
        # 不将原始错误体或请求头写日志，避免泄露凭据。
        detail = ""
        try:
            body = json.loads(exc.read(16384))
            detail = str(body.get("message") or body.get("error", {}).get("message", ""))
            detail = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", detail.replace(api_key, "[REDACTED]"))[:300]
        except (ValueError, AttributeError, TypeError):
            pass
        raise APIHTTPError(exc.code, detail) from None
    except (error.URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError("API 网络或响应异常，请求可能已计费；不自动重试") from None


def check_answer(response):
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if choice["finish_reason"] != "stop":
            return None, "incomplete_response"
        if not isinstance(content, str) or not content.strip():
            return None, "empty_content"
        if "<think>" in content or "</think>" in content:
            return None, "reasoning_in_content"
        blocks = re.findall(r"```(?:python|python3|py)\s*\n(.*?)```", content, re.S)
        if not blocks:
            return None, "missing_python_code"
        for code in blocks:
            compile(code, "<teacher>", "exec")  # 不执行教师代码。
        return content.strip(), "syntax_checked_pending_review"
    except SyntaxError:
        return None, "syntax_error"
    except (KeyError, IndexError, TypeError):
        return None, "invalid_response"


def append_journal(path, row):
    with JOURNAL_LOCK, path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def export_candidates(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("题库与运行清单不匹配")
    questions = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assignments, model = manifest["assignments"], manifest["model"]
    journal = output / "journal.jsonl"
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    latest = {e["id"]: e["event"] for e in events}
    uncertain = {identity for identity, event in latest.items() if event == "started"}
    records, statuses, usage = [], {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    completed = {e["id"]: e for e in events if e["event"] == "completed"}
    for question in questions:
        event = completed.get(question["id"])
        if not event:
            continue
        statuses[event["status"]] = statuses.get(event["status"], 0) + 1
        for name in usage:
            usage[name] += event["usage"].get(name, 0)
        response = json.loads((output / (question["id"] + ".response.json")).read_text(encoding="utf-8"))
        content, status = check_answer(response)
        if content is None:
            continue
        records.append({name: question[name] for name in ("id", "task_type", "language", "license", "group_id")})
        records[-1].update(source="siliconflow_teacher:" + model, split=assignments[question["id"]],
            teacher_model=response.get("model", model), status=status,
            verification={"syntax": True, "execution": False, "human_review": False},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question["prompt"]}, {"role": "assistant", "content": content}])
    candidates = output / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {"completed": len(completed), "candidates": len(records), "remaining": len(questions)-len(completed),
        "uncertain_ids": sorted(uncertain),
        "statuses": statuses, "usage": usage, "note": "仅语法检查，未经独立测试或人工审核，不是最终训练集；费用以平台账单为准"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def generate(source, output, limit=3, max_tokens=2048, timeout=120, api=None, workers=1, skip_uncertain=False):
    if workers not in (1, 2, 3, 4):
        raise ValueError("workers 必须为 1 到 4")
    if limit <= 0 or max_tokens <= 0 or timeout <= 0:
        raise ValueError("limit、max_tokens、timeout 必须为正数")
    key, model = credentials()
    source, output = Path(source), Path(output)
    questions = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    ids = [q["id"] for q in questions]
    if not questions or len(set(ids)) != len(ids):
        raise ValueError("题目为空或 id 重复")
    if any(q.get("source") != "pyforge_original_seed_v1" for q in questions):
        raise ValueError("仅允许原创种子题，禁止把算法验证集用于生成训练答案")
    splits = split_records(questions)
    assignments = {q["id"]: name for name, rows in splits.items() for q in rows}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "model": model,
        "max_tokens": max_tokens, "temperature": 0.2, "enable_thinking": False,
        "system": SYSTEM, "split_seed": 42, "assignments": assignments}
    manifest_path = output / "generation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("配置或题库与历史运行不一致，请使用新输出目录")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # 防止两个进程同时付费调用相同题目；异常退出后锁文件需人工确认再删除。
    lock = output / ".generation.lock"
    try:
        lock_stream = lock.open("x")
    except FileExistsError:
        raise ValueError("生成锁已存在，请确认没有运行中的生成任务") from None
    try:
        lock_stream.write(str(os.getpid()))
        lock_stream.close()
        journal = output / "journal.jsonl"
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
        finished = {e["id"] for e in events if e["event"] == "completed"}
        latest = {e["id"]: e["event"] for e in events}
        uncertain = {identity for identity, event in latest.items() if event == "started"}
        if uncertain and not skip_uncertain:
            raise ValueError("存在未确认结果的请求：{}；请核查账单与 journal，禁止自动重复付费".format(", ".join(sorted(uncertain))))
        pending = [q for q in questions if q["id"] not in finished and q["id"] not in uncertain][:limit]
        api = api or call_api
        def process_question(question):
            identity = question["id"]
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question["prompt"] + "\n\n" + question["answer_requirements"]}]
            payload = {"model": model, "messages": messages, "stream": False, "temperature": 0.2,
                       "max_tokens": max_tokens, "enable_thinking": False}
            append_journal(journal, {"event": "started", "id": identity, "time": datetime.now(timezone.utc).isoformat()})
            started_at = time.monotonic()
            try:
                response = api(key, payload, timeout)
            except APIHTTPError as exc:
                append_journal(journal, {"event": "http_error", "id": identity, "http_status": exc.status,
                    "time": datetime.now(timezone.utc).isoformat(), "note": "已收到 HTTP 错误；无答案；不自动重试，费用以平台账单为准"})
                raise
            # 先落盘完整原始响应，再记录完成；不将 reasoning_content 作为训练回答。
            (output / (identity + ".response.json")).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
            content, status = check_answer(response)
            append_journal(journal, {"event": "completed", "id": identity, "status": status,
                "usage": response.get("usage", {}), "response_model": response.get("model"),
                "elapsed_seconds": round(time.monotonic()-started_at, 2), "time": datetime.now(timezone.utc).isoformat()})
            print("{}: {}".format(identity, status), flush=True)
        # 每批最多 workers 个请求；一批失败后不再发起后续批次。
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                futures = [pool.submit(process_question, q) for q in pending[offset:offset+workers]]
                failures = []
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(exc)
                if failures:
                    raise failures[0]
        return export_candidates(source, output)
    finally:
        if not lock_stream.closed:
            lock_stream.close()
        lock.unlink(missing_ok=True)
