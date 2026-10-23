# 项目协作规则

## 文件命名

- Python 源码、测试、脚本与配置文件使用英文命名。
- 说明文档使用中文文件名；`README.md` 和 `AGENTS.md` 保留约定名称。
- 当前开发环境使用 Python 3.12；本地 RTX 4060 用于小规模测试，RTX 4080 SUPER 用于正式训练。

## 数据编写

- 默认由 Codex 直接编写、检查训练答案，使用 `datasets/`；硅基流动数据仅保留为历史版本。
- 当前三个集合固定为 `datasets/train.jsonl`、`datasets/validation.jsonl`、`datasets/test.jsonl`。来源、检查记录、算法评估与旧数据分别放入 sources、metadata、benchmarks、archive 子目录。
- 旧加法示例已清理；按用户要求保留旧版硅基流动数据和原始输出。
- 除非用户重新明确要求，不调用硅基流动或其他付费模型 API 生成答案。
- 答案必须记录真实来源；自动执行检查不能标记为人工审核。

## Git 提交

- 每完成一个独立功能，完成相应检查后主动创建一次 Git 提交，无需再次询问；仅提交该功能相关变更。
- 每次 Git 提交的 message 必须以 `【code by codex GPT-6.1】` 开头，前缀之后再填写本次变更说明。
- 示例：`【code by codex GPT-6.1】添加训练配置与数据校验模块`。
