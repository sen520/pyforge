"""生成自有合成冒烟数据；仅验证流程，不用于证明能力提升。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for i in range(60):
            row = {"id": "demo-{:03d}".format(i), "task_type": "code_generation", "language": "zh",
                "source": "self_authored_demo", "license": "self_owned", "group_id": "demo-family-{}".format(i // 3),
                "messages": [{"role": "user", "content": "用 Python 编写 add_{}(x)，返回 x 加 {}。".format(i, i)},
                             {"role": "assistant", "content": "```python\ndef add_{}(x):\n    return x + {}\n```".format(i, i)}]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
