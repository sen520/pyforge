# PyForge

面向个人 GPU 的 Python 编程助手微调项目。目标是在 **NVIDIA RTX 4080 SUPER（16 GB 显存）** 上，微调一个可在本地使用的小型模型，改善其中文需求理解、Python 代码生成、代码解释、错误修复和单元测试生成表现。

> 当前状态：已实现数据校验、分组划分、QLoRA 训练入口与单次推理入口。已在原生 Windows、Python 3.12、RTX 4060 上完成 10 步真实 GPU 训练、完整断点恢复和适配器重载推理。已接入 20 题、100 用例的 Docker 算法评估器。

## 本地开始

本地 RTX 4060 Laptop GPU（8 GB）用于冒烟测试，RTX 4080 SUPER（16 GB）用于正式实验。请先阅读 [本地开发指南](docs/本地开发指南.md)。

在已激活的 Python 3.12 虚拟环境中：

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
pyforge validate datasets/train.jsonl
```

训练前需安装支持本机驱动的 CUDA 版 PyTorch，以及训练扩展依赖，详见指南。安装完成后：

```powershell
pyforge doctor
pyforge train --config configs/smoke_4060.json --data datasets --output outputs/smoke-001 --offline
pyforge chat --run outputs/smoke-001 --offline --prompt "用 Python 编写一个判断偶数的函数"
```

上述离线命令需先完成下方的模型下载；移除 `--offline` 后，缓存缺失时会自动下载。训练、暂停与恢复详见 [训练运行指南](docs/训练运行指南.md)。旧的加法示例数据已删除，冒烟测试也使用当前问答数据。

三个训练用文件统一放在同一目录：`datasets/train.jsonl`、`datasets/validation.jsonl`、`datasets/test.jsonl`。题目来源放在 `datasets/sources/`，检查记录放在 `datasets/metadata/`，独立算法评估题放在 `datasets/benchmarks/`，旧版硅基流动数据放在 `datasets/archive/`。这些数据均提交 Git；模型、原始输出与下载缓存不提交。

## 训练、验证与测试

完整命令见[训练运行指南的命令速查](docs/训练运行指南.md#训练验证与测试命令速查)，包括连续多轮训练、断点恢复、验证损失查看、单题推理、Docker 算法评估与项目测试。

```powershell
# 检查三个数据文件的格式
pyforge validate datasets/train.jsonl
pyforge validate datasets/validation.jsonl
pyforge validate datasets/test.jsonl

# 使用新目录训练；5 轮需先设置 epochs=5、max_steps=-1
pyforge train --config configs/smoke_4060.json --data datasets --output outputs/train-003 --offline

# 验证在训练时自动执行；查看最佳验证损失和检查点
Get-Content outputs/train-003/metrics.json

# 测试适配器的单次回答
pyforge chat --run outputs/train-003 --offline --max-new-tokens 512 --prompt "编写保持原顺序的列表去重函数，并给出边界条件测试。"

# 测试项目代码
python -m unittest discover -s tests -v
```

`validate` 是数据校验，不是模型评估。历史 5 道留出题已存档，以下脚本默认读取归档数据，可用于复跑 train-002；不适用于新扩充的 123 道测试题：

```powershell
python scripts/evaluate_holdout.py generate --run outputs/train-002 --output outputs/holdout-train-002-v1
python scripts/evaluate_holdout.py score --run outputs/train-002 --output outputs/holdout-train-002-v1
```

脚本分别生成基础模型和微调模型的答案；输出目录须为新目录。GPU、镜像准备、报告及适用范围见训练指南。独立算法题执行评估仍需另行准备预测文件。

已完成 train-002 的 5 题留出集对比：基础模型 0/5，微调模型 1/5；仅代表有限用例结果，详见[留出集测试报告](docs/留出集测试报告.md)。

## 当前训练数据

当前默认数据位于 `datasets/`：训练 **2,292 条**、验证 **129 条**、新留出测试 **123 条**。新增 2,449 道 doocs 题目及其已有 Python 参考答案，保留真实来源；保留原有 90 条训练与 5 条验证问答。旧的 5 条测试已用于分析，存档到 `datasets/archive/codex_v1/`，不再充当新的独立留出集。

原始 Codex 答案有 Docker 执行记录；新增上游答案仅通过语法、静态依赖、完整对话长度检查，**尚未逐题执行验证**。原有 20 道可执行算法评估题排除在新增训练/验证/测试划分外。详情见[数据集说明](datasets/数据集说明.md)与[训练数据扩充说明](docs/训练数据扩充说明.md)。本轮不调用模型 API。

```powershell
# 4060：使用 2048 token 配置，以免 512 token 过滤大部分新增问答
pyforge train --config configs/train_4060.json --data datasets --output outputs/train-doocs-001 --offline
# 4080 SUPER：需先准备对应 1.5B 模型
pyforge train --config configs/train_4080.json --data datasets --output outputs/train-doocs-4080-001 --offline
```

## 提前下载模型

在已激活的 Python 3.12 虚拟环境中安装下载工具：

```powershell
python -m pip install "huggingface_hub>=0.34,<1"
```

本地 RTX 4060 冒烟测试使用 0.5B 模型：

```powershell
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --cache-dir D:/sen/pyforge/models/hub
```

RTX 4080 SUPER 正式实验使用 1.5B 模型，可按需下载：

```powershell
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --cache-dir D:/sen/pyforge/models/hub
```

上述路径对应当前仓库位置；如果项目放在其他位置，请替换为该项目下 `models/hub` 的绝对路径。训练和推理代码自动使用项目内的这一缓存目录，无需设置环境变量。手动下载使用 `--cache-dir`，以保持与代码一致的缓存布局。

下载中断后重新执行相同命令即可复用已完成的缓存。模型文件不提交到 Git。下载工具与参数说明见 [Hugging Face 官方 CLI 文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

## 项目目标

- 根据中文或英文需求生成清晰、可运行的 Python 代码。
- 解释现有代码，定位常见异常并给出修复方案。
- 为函数生成覆盖正常情况和边界条件的 pytest 测试。
- 在固定测试集上对比原始模型与微调模型，用执行结果验证收益。
- 保存可复现的训练配置、数据版本和 LoRA 适配器。

首版聚焦单文件、函数级任务与 Python 标准库。复杂仓库修改、联网检索、IDE 插件、自动执行用户代码和强化学习训练不属于首版范围。微调用于调整任务表现与回答风格，效果必须通过评估确认。

## 初始技术方案

| 项目 | 初始选择 | 说明 |
| --- | --- | --- |
| 基础模型 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 已具备编程与对话能力，适合作为小规模实验起点 |
| 训练方式 | SFT + QLoRA | 4-bit 加载基础权重，仅训练 LoRA 参数 |
| 计算精度 | BF16 | 启动时检测环境支持，不支持时显式报错 |
| 训练框架 | PyTorch、Transformers Trainer、PEFT、Accelerate、bitsandbytes | 训练扩展版本已固定，并通过本机 4060 冒烟验证 |
| 运行环境 | NVIDIA CUDA | 原生 Windows 已完成 4060 验证，其他环境需自检 |
| 交互方式 | 本地命令行 | 后续按需增加 API 或页面 |

这是一套起始实验方案，并非已验证的最优模型组合。首版通过验收后，再评估扩大模型规模的收益与显存成本。

## 硬件与环境

- GPU：RTX 4080 SUPER，16 GB 显存；训练前关闭其他占用 GPU 的应用。
- 主机内存：建议至少 32 GB；磁盘建议预留 50 GB，用于模型缓存、数据与检查点。实际需求随数据和保存策略变化。
- Python：支持 3.11 / 3.12，当前开发目标为 3.12；训练扩展已固定实测版本，完整版本记录见运行清单。
- 先使用 `nvidia-smi` 检查驱动与 GPU 状态，再验证 PyTorch 能否访问 CUDA。该命令显示的 CUDA 版本不能代替 PyTorch 运行时兼容性检查。

CUDA / PyTorch 组合须通过本机自检；4060 的训练、保存、恢复与推理已验证，4080 SUPER 正式配置尚未实测。

## 计划工作流

1. 建立环境，验证 GPU、量化加载、一次前向与反向传播。
2. 收集有明确使用权限的数据，完成清洗、去重和数据划分。
3. 对原始模型运行固定基线评估。
4. 使用小样本完成训练、保存、重新加载和推理的冒烟验证。
5. 执行正式训练，根据验证集选择检查点。
6. 在冻结测试集上比较基础模型与适配器模型。
7. 输出适配器、训练清单、评估报告和本地推理入口。

历史硅基流动数据保留在 `datasets/archive/teacher_v1/`，不再作为当前默认训练集；不要与新数据直接合并，以免重复题或分组泄漏。

## 文档导航

- [训练运行指南](docs/训练运行指南.md)：训练、离线加载、暂停恢复、输出文件与配置说明。
- [训练验证记录](docs/训练验证记录.md)：4060 实测环境、损失与显存记录。

- [答案编写与检查指南](docs/答案编写与检查指南.md)：无需模型 API 的答案来源、离线构建和 Docker 检查流程。

- [历史教师数据生成指南](docs/教师数据生成指南.md)：旧版硅基流动工具，仅保留历史说明，不作为当前流程。

- [算法评估运行指南](docs/算法评估运行指南.md)：20 道题、100 个用例的可执行子集与 Docker 评估命令。

- [算法验证集说明](docs/算法验证集说明.md)：从 doocs/leetcode 提取的独立验证候选集、复现方式及评估边界。

- [题库说明](docs/题库说明.md)：首批 100 道原创种子题及教师模型作答流程；题库位于 `datasets/sources/seed_questions.jsonl`。

- [开发文档](docs/开发文档.md)：架构、模块职责、配置、训练与开发里程碑。
- [数据与评估规范](docs/数据与评估规范.md)：数据格式、质量规则、划分与验收标准。

## 计划目录

```text
pyforge/
├── README.md
├── docs/
├── configs/             # 4060 冒烟与 4080 训练配置
├── src/pyforge/         # 数据、编码、训练、推理与 CLI
├── scripts/             # 答案编写、数据构建与检查
├── tests/               # 数据与标签逻辑测试
├── datasets/            # train / validation / test，以及来源、评估题和归档
├── data/                # 上游仓库下载缓存，不提交
└── outputs/             # 训练产物、临时构建和原始响应，不提交
```

## 资料与许可

- [基础模型官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [PEFT 量化训练文档](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Transformers bitsandbytes 文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [NVIDIA RTX 4080 系列规格](https://www.nvidia.com/en-my/geforce/graphics-cards/40-series/rtx-4080-family/)

发布前需确认基础模型、数据集和训练产物的许可义务。仓库目前尚未选定项目代码许可证；公开代码不代表自动获得任意使用授权。
