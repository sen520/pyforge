import argparse
import json
import sys

from .data import prepare, read_records


def main():
    parser = argparse.ArgumentParser(description="PyForge 数据清洗配置模型训练与评估")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查 CUDA 与 BF16")
    comparison = commands.add_parser("benchmark", help="一键数据清洗测试、验证、基础/微调对比与 Excel 导出")
    comparison.add_argument("--run", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--data", default="datasets/data_cleaning")
    comparison.add_argument("--reuse-predictions", help="复用已有四份完整预测，不加载模型")
    comparison.add_argument("--max-new-tokens", type=int, default=512)
    comparison.add_argument("--node", help="覆盖 Codex 附带 Node 可执行文件路径")
    comparison.add_argument("--artifact-modules", help="覆盖附带 node_modules 路径")
    validate = commands.add_parser("validate", help="校验 JSONL 数据")
    validate.add_argument("source")
    split = commands.add_parser("prepare", help="按 group_id 划分数据")
    split.add_argument("source")
    split.add_argument("destination")
    split.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="执行 QLoRA 训练")
    training.add_argument("--config", required=True)
    training.add_argument("--data", default="datasets/data_cleaning")
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
        elif args.command == "benchmark":
            from .data_cleaning_benchmark import benchmark
            comparison = benchmark(args.run, args.output, args.data, args.reuse_predictions,
                                   args.max_new_tokens, args.node, args.artifact_modules)
            result = {split: payload["summary"] for split, payload in comparison.items()}
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
