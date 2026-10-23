import argparse
import json
import sys

from .data import prepare, read_records


def main():
    parser = argparse.ArgumentParser(description="PyForge Python 编程助手微调")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查 CUDA 与 BF16")
    evaluation = commands.add_parser("evaluate", help="Docker 隔离评估算法回答")
    evaluation.add_argument("--suite", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--report", required=True)
    evaluation.add_argument("--image", default="python:3.12-slim")
    evaluation.add_argument("--timeout", type=float, default=5)
    validate = commands.add_parser("validate", help="校验 JSONL 数据")
    validate.add_argument("source")
    split = commands.add_parser("prepare", help="按 group_id 划分数据")
    split.add_argument("source")
    split.add_argument("destination")
    split.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="执行 QLoRA 训练")
    training.add_argument("--config", required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--resume")
    chat = commands.add_parser("chat", help="加载适配器，执行一次推理")
    chat.add_argument("--run", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = {"valid_records": len(read_records(args.source))}
        elif args.command == "prepare":
            result = prepare(args.source, args.destination, args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.suite, args.predictions, args.report, args.image, args.timeout)
        else:
            from .training import chat as generate, gpu_check, train
            if args.command == "doctor":
                result = gpu_check()
            elif args.command == "train":
                result = train(args.config, args.data, args.output, args.resume)
            else:
                if args.max_new_tokens <= 0:
                    raise ValueError("max-new-tokens 必须为正整数")
                print(generate(args.run, args.prompt, args.max_new_tokens))
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print("错误：{}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
