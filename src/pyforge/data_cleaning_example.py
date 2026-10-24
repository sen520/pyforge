"""使用训练后的数据清洗模型：中文需求 → JSON 配置 → 本地清洗结果。

在项目根目录运行：
    python -m pyforge.data_cleaning_example --run outputs/data-cleaning-002
只生成配置：
    python -m pyforge.data_cleaning_example --prompt '按“积分”降序排列。' --config-only
"""

import argparse
import json
from pathlib import Path

from pyforge.data_cleaning import SYSTEM_PROMPT, execute_config, parse_config
from pyforge.paths import PROJECT_ROOT


DEFAULT_RUN = PROJECT_ROOT / 'outputs/data-cleaning-002'
DEFAULT_PROMPT = '先删除“邮箱”为null的行，再按“邮箱”去重保留最后一条，最后按“积分”降序排列。'
DEMO_ROWS = [
    {'邮箱': 'a@example.com', '积分': 50},
    {'邮箱': None, '积分': 99},
    {'邮箱': 'b@example.com', '积分': 80},
    {'邮箱': 'a@example.com', '积分': 90},
    {'邮箱': 'c@example.com', '积分': 60},
]


def generate_config(prompt, run=DEFAULT_RUN, max_new_tokens=256):
    """加载基础模型和训练后的 LoRA；system 提示由训练清单自动恢复。"""
    if not isinstance(prompt, str) or not prompt.strip() or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('需求必须为非空字符串，max_new_tokens 必须为正整数')
    run = Path(run)
    manifest_path = run / 'run_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'找不到 {manifest_path}，请指定实际存在的训练目录')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('system_prompt') != SYSTEM_PROMPT:
        raise ValueError('该运行不是当前数据清洗配置模型，请指定匹配的训练目录')
    if not (run / 'adapter/adapter_model.safetensors').is_file():
        raise ValueError('找不到 adapter/adapter_model.safetensors，请先完成训练')

    # 延迟导入：查看 --help 或导入示例不需要加载 GPU 依赖。
    from pyforge.training import chat

    response = chat(run, prompt, max_new_tokens=max_new_tokens, offline=True)
    try:
        config = parse_config(response)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'模型回答不是合法清洗配置，未执行清洗。原始回答：\n{response}') from exc
    return response, config


def clean_rows(prompt, rows, run=DEFAULT_RUN, max_new_tokens=256):
    """数据行只在本地执行器中处理；只将中文需求发送给本地模型。"""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('输入数据必须是 JSON 对象数组，例如 [{"邮箱":"a@example.com","积分":90}]')
    response, config = generate_config(prompt, run, max_new_tokens)
    # 仅解释去重、空值处理、筛选、排序，不执行模型返回的 Python 代码。
    cleaned = execute_config(config, rows)
    return {'prompt': prompt, 'raw_response': response, 'config': config,
            'input_count': len(rows), 'output_count': len(cleaned), 'rows': cleaned}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN, help='完成训练的运行目录，默认 outputs/data-cleaning-002')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='中文清洗需求，列名须与输入数据一致')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--input-json', type=Path, help='本地 JSON 对象数组；不指定时使用内置演示数据')
    mode.add_argument('--config-only', action='store_true', help='仅生成并校验配置，不清洗数据')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if not args.prompt.strip() or args.max_new_tokens <= 0:
        parser.error('需求不能为空，max-new-tokens 必须为正数')
    try:
        if args.config_only:
            _, result = generate_config(args.prompt, args.run, args.max_new_tokens)
        else:
            rows = (json.loads(args.input_json.read_text(encoding='utf-8-sig'))
                    if args.input_json else DEMO_ROWS)
            result = clean_rows(args.prompt, rows, args.run, args.max_new_tokens)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'使用示例失败：{exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
