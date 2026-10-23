"""模型响应的 Docker 隔离执行评估，不在宿主执行候选代码。"""

import hashlib
import json
import math
import re
import uuid
import subprocess
import tempfile
import time
from pathlib import Path


WORKER = r'''
import contextlib, io, json, typing
from pathlib import Path
job = json.loads(Path('/job/job.json').read_text())
namespace = {name: getattr(typing, name) for name in ('List', 'Optional', 'Dict', 'Tuple', 'Set')}
namespace['__name__'] = '__candidate__'
outputs = []
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(job['code'], '<candidate>', 'exec'), namespace)
    for args in job['args']:
        result = getattr(namespace['Solution'](), job['method'])(*args)
        outputs.append(result)
print(json.dumps(outputs, allow_nan=False))
'''


def extract_code(response):
    if not isinstance(response, str) or not response.strip():
        raise ValueError("回答为空")
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", response, re.S)
    if blocks:
        python_blocks = [code for language, code in blocks if language.strip().lower() in ("python", "python3", "py")]
        if len(python_blocks) != 1 or len(blocks) != 1:
            raise ValueError("要求唯一 Python 代码块")
        code = python_blocks[0]
    elif "```" in response:
        raise ValueError("代码块未闭合")
    else:
        code = response
    compile(code, "<candidate>", "exec")  # 仅检查语法，不执行。
    return code


def compare(actual, expected, mode):
    if mode == "exact":
        # bool 不能冒充 int，递归保留 JSON 类型约束。
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(compare(a, b, "exact") for a, b in zip(actual, expected))
        return actual == expected
    if mode == "unordered":
        return (isinstance(actual, list) and all(isinstance(x, str) for x in actual)
                and sorted(actual) == sorted(expected))  # 保留重复计数。
    if mode == "unordered_nested":
        if not isinstance(actual, list) or not all(isinstance(x, list) and all(type(n) is int for n in x) for x in actual):
            return False
        return sorted(sorted(x) for x in actual) == sorted(sorted(x) for x in expected)
    raise ValueError("未知比较规则：" + mode)


def docker_environment(image):
    try:
        info = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=15)
        if info.returncode or info.stdout.strip() != "linux":
            raise RuntimeError("需要可访问的 Docker Linux 容器引擎")
        result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError("缺少镜像，请执行 docker pull " + image)
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Docker 不可用：" + str(exc)) from exc


def run_isolated(code, problem, image_id, timeout=5, include_details=False):
    name = "pyforge-eval-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="pyforge-eval-") as directory:
        directory = Path(directory)
        (directory / "worker.py").write_text(WORKER, encoding="utf-8")
        (directory / "job.json").write_text(json.dumps({"code": code, "method": problem["method"],
            "args": [case["args"] for case in problem["tests"]]}), encoding="utf-8")
        command = ["docker", "run", "--rm", "--pull=never", "--name", name, "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
            "--memory=256m", "--memory-swap=256m", "--cpus=1", "--pids-limit=32", "--log-driver=none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir=/tmp",
            "--mount", "type=bind,source={},target=/job,readonly".format(directory),
            image_id, "python", "-I", "-B", "/job/worker.py"]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        return {"status": "timeout"}
                    if stdout.tell() + stderr.tell() > 65536:
                        return {"status": "output_limit"}
                    time.sleep(0.02)
                stdout.seek(0)
                stderr.seek(0)
                data = stdout.read(65537)
                error_data = stderr.read(65537)
                if len(data) + len(error_data) > 65536:
                    return {"status": "output_limit"}
                if process.returncode in (125, 126, 127):
                    raise RuntimeError("容器启动失败：" + str(process.returncode))
                if process.returncode:
                    result = {"status": "runtime_error"}
                    if include_details:
                        result['stderr'] = error_data.decode('utf-8', errors='replace')[-4000:]
                    return result
                try:
                    values = json.loads(data)
                except (ValueError, UnicodeError):
                    return {"status": "invalid_output"}
                if not isinstance(values, list) or len(values) != len(problem["tests"]):
                    return {"status": "invalid_output"}
                passed = sum(compare(value, case["expected"], problem["comparator"]) for value, case in zip(values, problem["tests"]))
                result = {"status": "passed" if passed == len(values) else "wrong_answer", "passed_cases": passed, "total_cases": len(values)}
                if include_details:
                    result['case_results'] = [compare(value, case['expected'], problem['comparator']) for value, case in zip(values, problem['tests'])]
                return result
            finally:
                cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    raise RuntimeError("容器清理失败：" + name)


def evaluate(suite_path, predictions_path, report_path, image="python:3.12-slim", timeout=5):
    suite_path, predictions_path, report_path = map(Path, (suite_path, predictions_path, report_path))
    if report_path.exists():
        raise ValueError("报告已存在，拒绝覆盖")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("超时必须为正数")
    problems = [json.loads(line) for line in suite_path.read_text(encoding="utf-8").splitlines()]
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    ids = [p["id"] for p in problems]
    if not ids or len(set(ids)) != len(ids) or not all(p.get("execution_ready") and p.get("tests") for p in problems):
        raise ValueError("题库必须非空、ID 唯一且包含可执行测试")
    answers = {}
    for row in predictions:
        if row["id"] in answers or row["id"] not in ids:
            raise ValueError("重复或未知预测 ID：" + row["id"])
        answers[row["id"]] = row["response"]
    image_id = docker_environment(image)
    results = []
    for problem in problems:
        result = {"id": problem["id"]}
        if problem["id"] not in answers:
            result["status"] = "missing_prediction"
        else:
            try:
                code = extract_code(answers[problem["id"]])
            except SyntaxError:
                result["status"] = "syntax_error"
            except ValueError:
                result["status"] = "extraction_error"
            else:
                result.update(run_isolated(code, problem, image_id, timeout))
        results.append(result)
    passed = sum(r["status"] == "passed" for r in results)
    report = {"total": len(problems), "passed": passed, "pass_at_1": passed / len(problems),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "execution_mode": "docker", "image_id": image_id, "timeout_seconds_per_problem": timeout, "results": results,
        "scope": "仅本地编写的测试；不等于官方隐藏测试通过率"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
