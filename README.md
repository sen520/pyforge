# PyForge

面向个人 GPU 的 Python 编程助手微调项目。目标是在 **NVIDIA RTX 4080 SUPER（16 GB 显存）** 上，微调一个可在本地使用的小型模型，改善其中文需求理解、Python 代码生成、代码解释、错误修复和单元测试生成表现。

> 当前状态：已实现数据校验、分组划分、QLoRA 训练入口与单次推理入口。数据和标签逻辑已有自动化测试；真实 GPU 训练与适配器重载尚待在 Python 3.12 环境验证。隔离代码评估器尚未实现。

## 本地开始

本地 RTX 4060 Laptop GPU（8 GB）用于冒烟测试，RTX 4080 SUPER（16 GB）用于正式实验。请先阅读 [本地开发指南](docs/本地开发指南.md)。

在已激活的 Python 3.12 虚拟环境中：

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/make_demo_data.py data/demo.jsonl
pyforge validate data/demo.jsonl
pyforge prepare data/demo.jsonl data/prepared
```

训练前需安装支持本机驱动的 CUDA 版 PyTorch，以及训练扩展依赖，详见指南。安装完成后：

```powershell
pyforge doctor
pyforge train --config configs/smoke_4060.json --data data/prepared --output outputs/smoke-001
pyforge chat --run outputs/smoke-001 --prompt "用 Python 编写一个判断偶数的函数"
```

合成示例数据只验证管线，不代表真实训练集或有效能力评估。首次训练会下载基础模型，需要网络与足够磁盘空间。

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
| 计算精度 | BF16 | 启动时检测环境支持，失败时显式报错或使用验证过的 FP16 配置 |
| 训练框架 | PyTorch、Transformers Trainer、PEFT、Accelerate、bitsandbytes | 当前依赖范围尚非实测锁定版本；首版直接使用 Trainer |
| 运行环境 | 优先 WSL2 / Linux + NVIDIA CUDA | 原生 Windows 作为后续兼容验证项 |
| 交互方式 | 本地命令行 | 后续按需增加 API 或页面 |

这是一套起始实验方案，并非已验证的最优模型组合。首版通过验收后，再评估扩大模型规模的收益与显存成本。

## 硬件与环境

- GPU：RTX 4080 SUPER，16 GB 显存；训练前关闭其他占用 GPU 的应用。
- 主机内存：建议至少 32 GB；磁盘建议预留 50 GB，用于模型缓存、数据与检查点。实际需求随数据和保存策略变化。
- Python：支持 3.11 / 3.12，当前开发目标为 3.12；GPU 环境验证后再提交实测依赖锁定文件。
- 先使用 `nvidia-smi` 检查驱动与 GPU 状态，再验证 PyTorch 能否访问 CUDA。该命令显示的 CUDA 版本不能代替 PyTorch 运行时兼容性检查。

CUDA / PyTorch 组合须通过本机自检；训练命令已实现，但尚未完成真实 GPU 验证。

## 计划工作流

1. 建立环境，验证 GPU、量化加载、一次前向与反向传播。
2. 收集有明确使用权限的数据，完成清洗、去重和数据划分。
3. 对原始模型运行固定基线评估。
4. 使用小样本完成训练、保存、重新加载和推理的冒烟验证。
5. 执行正式训练，根据验证集选择检查点。
6. 在冻结测试集上比较基础模型与适配器模型。
7. 输出适配器、训练清单、评估报告和本地推理入口。

## 文档导航

- [算法验证集说明](docs/算法验证集说明.md)：从 doocs/leetcode 提取的独立验证候选集、复现方式及评估边界。

- [题库说明](docs/题库说明.md)：首批 100 道原创种子题及教师模型作答流程；题库位于 `datasets/seed_questions.jsonl`。

- [开发文档](docs/开发文档.md)：架构、模块职责、配置、训练与开发里程碑。
- [数据与评估规范](docs/数据与评估规范.md)：数据格式、质量规则、划分与验收标准。

## 计划目录

```text
pyforge/
├── README.md
├── docs/
├── configs/             # 4060 冒烟与 4080 训练配置
├── src/pyforge/         # 数据、编码、训练、推理与 CLI
├── scripts/             # 合成冒烟数据生成器
├── tests/               # 数据与标签逻辑测试
├── data/                # 本地数据，不提交原始语料（待创建）
└── outputs/             # 训练产物与报告，不提交模型权重（待创建）
```

## 资料与许可

- [基础模型官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [PEFT 量化训练文档](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Transformers bitsandbytes 文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [NVIDIA RTX 4080 系列规格](https://www.nvidia.com/en-my/geforce/graphics-cards/40-series/rtx-4080-family/)

发布前需确认基础模型、数据集和训练产物的许可义务。仓库目前尚未选定项目代码许可证；公开代码不代表自动获得任意使用授权。
