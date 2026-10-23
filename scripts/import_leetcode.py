"""从本地 doocs 仓库提取独立验证候选题集。"""

import argparse
import json

from pyforge.leetcode import build_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/sources/doocs-leetcode")
    parser.add_argument("--output", default="data/validation/doocs-leetcode")
    parser.add_argument("--overlap-rules", default="configs/leetcode_overlap.json")
    args = parser.parse_args()
    manifest = build_validation(args.source, args.output, args.overlap_rules)
    print(json.dumps({key: manifest[key] for key in ("source_revision", "count", "difficulty")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
