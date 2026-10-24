"""一条命令完成数据清洗验证/测试、基础模型对比和 Excel 导出。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pyforge.data_cleaning_benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='必须是新目录')
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/data_cleaning')
    parser.add_argument('--reuse-predictions', type=Path, help='复用已有四份预测、生成清单和题面，不加载模型')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='新生成时的上限；复用时沿用并核对原生成参数')
    parser.add_argument('--node', type=Path, help='覆盖 Codex 附带 Node 可执行文件路径')
    parser.add_argument('--artifact-modules', type=Path, help='覆盖附带 node_modules 路径')
    args = parser.parse_args()
    results = benchmark(args.run, args.output, args.data, args.reuse_predictions,
                        args.max_new_tokens, args.node, args.artifact_modules)
    for split, comparison in results.items():
        summary = comparison['summary']
        print(f'{split}: 严格基础 {summary["strict_before"]["correct"]}/{summary["total"]}，'
              f'微调 {summary["strict_after"]["correct"]}/{summary["total"]}；'
              f'诊断基础 {summary["diagnostic_before"]["correct"]}/{summary["total"]}，'
              f'微调 {summary["diagnostic_after"]["correct"]}/{summary["total"]}')
    print('Excel：', args.output.resolve() / '模型效果对比.xlsx')


if __name__ == '__main__':
    main()
