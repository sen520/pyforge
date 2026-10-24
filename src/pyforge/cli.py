import argparse
import json
import sys

from .data import prepare, read_records


def main():
    parser = argparse.ArgumentParser(description="PyForge Python 编程助手微调")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查 CUDA 与 BF16")
    generation = commands.add_parser("generate", help="使用硅基流动生成教师问答候选数据")
    generation.add_argument("--source", default="datasets/leetcode/sources/seed_questions.jsonl")
    generation.add_argument("--output", default="outputs/teacher_v1")
    generation.add_argument("--limit", type=int, default=3)
    generation.add_argument("--max-tokens", type=int, default=2048)
    generation.add_argument("--workers", type=int, default=1)
    generation.add_argument("--skip-uncertain", action="store_true", help="跳过结果未确认的请求，仅生成其他题目")
    generation.add_argument("--timeout", type=int, default=120)
    evaluation = commands.add_parser("evaluate", help="Docker 隔离评估算法回答")
    evaluation.add_argument("--suite", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--report", required=True)
    evaluation.add_argument("--image", default="python:3.12-slim")
    evaluation.add_argument("--timeout", type=float, default=5)
    comparison = commands.add_parser("benchmark", help="一键生成基础/微调答案、执行相同测试并输出对照表")
    comparison.add_argument("--run", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--suite", help="默认使用保留的 20 道可执行算法题")
    comparison.add_argument("--data", default="datasets/leetcode")
    comparison.add_argument("--max-new-tokens", type=int, default=1024)
    comparison.add_argument("--image", default="python:3.12-slim")
    comparison.add_argument("--timeout", type=float, default=10)
    validate = commands.add_parser("validate", help="校验 JSONL 数据")
    validate.add_argument("source")
    split = commands.add_parser("prepare", help="按 group_id 划分数据")
    split.add_argument("source")
    split.add_argument("destination")
    split.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="执行 QLoRA 训练")
    training.add_argument("--config", required=True)
    training.add_argument("--data", default="datasets/leetcode")
    training.add_argument("--output", required=True)
    training.add_argument("--resume")
    training.add_argument("--offline", action="store_true", help="仅使用项目中的模型缓存")
    training.add_argument("--stop-after", type=int, help="达到指定全局步数后保存并提前停止，便于验证恢复")
    chat = commands.add_parser("chat", help="加载适配器，执行一次推理")
    chat.add_argument("--run", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-new-tokens", type=int, default=512)
    chat.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = {"valid_records": len(read_records(args.source))}
        elif args.command == "prepare":
            result = prepare(args.source, args.destination, args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.suite, args.predictions, args.report, args.image, args.timeout)
        elif args.command == "benchmark":
            from .benchmark import benchmark, DEFAULT_SUITE
            result = benchmark(args.run, args.output, args.suite or DEFAULT_SUITE, args.data,
                args.max_new_tokens, args.image, args.timeout)
        elif args.command == "generate":
            from .generation import generate
            result = generate(args.source, args.output, args.limit, args.max_tokens, args.timeout,
                              workers=args.workers, skip_uncertain=args.skip_uncertain)
        else:
            from .training import chat as generate, gpu_check, train
            if args.command == "doctor":
                result = gpu_check()
            elif args.command == "train":
                result = train(args.config, args.data, args.output, args.resume, args.offline, args.stop_after)
            else:
                if args.max_new_tokens <= 0:
                    raise ValueError("max-new-tokens 必须为正整数")
                print(generate(args.run, args.prompt, args.max_new_tokens, args.offline))
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print("错误：{}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
