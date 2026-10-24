# PyForge

面向个人 NVIDIA GPU 的 SFT + QLoRA 微调项目，使用 Python 3.12、Transformers Trainer、PEFT 和 bitsandbytes。当前任务是中文需求转 JSON 数据清洗配置，保留此前的 Python/LeetCode 实验。工作目录为 `E:\code\github\pyforge`，RTX 4080 SUPER（16 GB）用于 1.5B 训练，RTX 4060 用于小规模测试。

当前已在原生 Windows、RTX 4060 上完成 0.5B 模型的 2 轮训练（1,146 步），支持离线加载、完整恢复、最佳检查点选择、推理、批量评估预测、日志绘图和轻量实验归档。训练完成不代表代码正确率达标。

RTX 4080 SUPER 的 1.5B 数据清洗训练、验证/测试预测和 Excel 导出也已在本地跑通。当前数据为训练 1,000 条、验证 200 条、测试 200 条；相关本地产物在 outputs/，不随 Git 提交。

## 脚本使用：训练、测试、验证与 Excel 模型对比

评估入口：[scripts/benchmark_data_cleaning.py](scripts/benchmark_data_cleaning.py)。按下面顺序训练、测试、验证；评估脚本读取已训练好的数据清洗模型，不重新训练。

### 1. 训练数据清洗模型

在 PowerShell 中执行：

```powershell
Set-Location E:\code\github\pyforge
.\.venv\Scripts\Activate.ps1
pyforge doctor
pyforge train --config configs/train_4080.json --data datasets/data_cleaning --output outputs/data-cleaning-002 --offline
```

使用 RTX 4080 SUPER 的 1.5B 配置，训练 2 轮、126 个优化器更新步；训练期间自动计算验证 loss。等待训练正常结束、run_status.json 为 completed 后，再执行下方评估命令。

训练输出目录必须是新目录。若 `data-cleaning-002` 已存在且训练完成，直接跳过此步；若需要新实验，请改为 `data-cleaning-003` 等新编号，并将下方评估命令的 `--run` 同步修改，不能覆盖已有目录。

### 2. 测试

已完成 `outputs/data-cleaning-002` 训练时，一条命令先运行测试、再运行验证，比较基础/微调模型并导出 Excel：

```powershell
Set-Location E:\code\github\pyforge
.\.venv\Scripts\Activate.ps1
python scripts/benchmark_data_cleaning.py --run outputs/data-cleaning-002 --output outputs/data-cleaning-comparison-001
```

执行顺序为测试集微调模型、测试集基础模型、验证集微调模型、验证集基础模型，各 200 条。每份预测分别计算严格评分和去代码块外壳的诊断评分，最后导出 Excel。

### 3. 验证

上面的脚本完成测试后，会自动继续验证集的微调和基础模型预测及评分，无需再执行一条验证命令。全部完成后生成 Excel，查看测试集与验证集各自的结果。训练过程中计算的验证 loss 与这里生成 JSON 答案的验证评分是两个指标。

### 4. 使用训练好的模型

使用示例位于 [src/pyforge/data_cleaning_example.py](src/pyforge/data_cleaning_example.py)。运行后加载基础模型及训练后的 LoRA，用中文需求生成 JSON 配置，再清洗内置的 5 行邮箱/积分数据：

```powershell
python -m pyforge.data_cleaning_example --run outputs/data-cleaning-002
```

只生成配置、不执行清洗：

```powershell
python -m pyforge.data_cleaning_example --run outputs/data-cleaning-002 --prompt '按“积分”降序排列。' --config-only
```

默认离线运行，仍需 CUDA 和本地模型缓存。支持通过 `--input-json` 读取自己的 JSON 对象数组，也可在 Python 中导入 `clean_rows()`；完整示例见[数据清洗数据集指南](docs/数据清洗数据集指南.md)。模型输出必须先通过配置校验，失败时停止，不自动修补或执行模型生成的 Python 代码。

### 已有预测时：跳过生成，只导出评分与 Excel

已有 `outputs/data-cleaning-eval-002` 中的四份完整预测时，可以直接复用，不再加载模型：

```powershell
python scripts/benchmark_data_cleaning.py --run outputs/data-cleaning-002 --reuse-predictions outputs/data-cleaning-eval-002 --output outputs/data-cleaning-comparison-002
```

### 参数和输出位置

| 参数 | 用途 |
| --- | --- |
| `--run` | 已完成训练的运行目录，须含 run_manifest.json、run_status.json 和 adapter/ |
| `--output` | 新的评估输出目录，不能已存在 |
| `--reuse-predictions` | 可选；指定已有四份预测、生成清单与题面所在目录，跳过模型预测 |
| `--data` | 可选；默认 datasets/data_cleaning |
| `--max-new-tokens` | 可选；新预测默认 512，复用时沿用原参数 |

完整运行示例生成 `outputs/data-cleaning-comparison-001/模型效果对比.xlsx`；复用示例生成 `outputs/data-cleaning-comparison-002/模型效果对比.xlsx`。同目录还包含 comparison.json、四份逐题评分和 pipeline_status.json。

输出目录必须是新目录，已存在时更换编号。Excel 为输出目录中的 `模型效果对比.xlsx`，包含效果汇总、验证集对比和测试集对比，可查看完整题面、参考答案、两边原始回答和改善/退步。严格评分与仅去掉代码块外壳的诊断评分分开显示，不能将格式合格率或配置匹配率当作执行正确率。

若尚未训练，先按[数据清洗数据集指南](docs/数据清洗数据集指南.md)完成环境准备与训练，再执行上面的评估。该任务全程离线、无需 Docker；Excel 使用本机 Codex 附带的 Node.js/artifact-tool。训练、预测和报告由你手动启动，完整命令与故障处理见指南。

## 安装环境

在 PowerShell 中先执行 `Set-Location E:\code\github\pyforge`，再激活自己准备的 Python 3.12 虚拟环境。实测 PyTorch 为 2.6.0+cu124；训练扩展版本固定在 pyproject.toml，安装源仍需单独指定。

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train,plot]"
pyforge doctor
python -m unittest discover -s tests -v
```

详见[本地开发指南](docs/本地开发指南.md)。Docker 用于执行模型代码，训练本身不需要 Docker。当前流程不需要 API 密钥；旧服务配置仅保存在忽略的 .env 中。

## 提前下载模型

```powershell
python -m pip install "huggingface_hub>=0.34,<1"
# 当前 4080 SUPER：1.5B，已完成本地数据清洗训练验证
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --cache-dir E:/code/github/pyforge/models/hub
# 可选：4060 小规模测试使用 0.5B
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --cache-dir E:/code/github/pyforge/models/hub
```

以上缓存路径对应当前工作目录；以后迁移时改为新项目下 models/hub 的绝对路径。代码优先使用该缓存；--offline 禁止联网，缺失版本报错。中断后重跑下载命令可复用缓存，模型权重不提交 Git。

## 当前数据

数据按任务整理：本轮旧数据已迁移到 `datasets/leetcode/`；新方向目录为 `datasets/data_cleaning/`，用于模型直接编写的新任务数据，详见[数据清洗数据集指南](docs/数据清洗数据集指南.md)。目录总览见[数据集目录](datasets/数据集说明.md)。历史 JSON 清单与实验报告保留原路径及哈希。

| 当前数据清洗集合 | 条数 | 用途 |
| --- | ---: | --- |
| datasets/data_cleaning/train.jsonl | 1,000 | 更新参数 |
| datasets/data_cleaning/validation.jsonl | 200 | 验证损失和开发评估 |
| datasets/data_cleaning/test.jsonl | 200 | 最终配置预测评估 |

完整作者原稿仍各 1,000 条，缩减前版本保留在 datasets/data_cleaning/archive/full_v1/。下表与后续算法训练命令属于保留的 LeetCode 流程，不与当前任务混用。

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| datasets/leetcode/train.jsonl | 2,292 | 更新参数 |
| datasets/leetcode/validation.jsonl | 129 | 验证损失和检查点选择 |
| datasets/leetcode/test.jsonl | 123 | 新问答留出集，尚无统一执行器 |

新增 2,449 条来自 doocs 既有 Python 参考答案，保留归属与许可，已做语法、静态依赖和完整对话长度检查，**未逐题执行验证**。另保留原有 90 条训练、5 条验证的已执行 Codex 数据；旧 5 条已分析测试题存档在 datasets/leetcode/archive/codex_v1/。

已完成 200 题抽样审计：189 题通过、11 题未适配；另修复训练/验证中 57 条题面主接口缺失，答案和划分不变。当前文件哈希见 datasets/leetcode/metadata/current_manifest.json，原始构建清单保留。详见[训练数据审计](docs/训练数据审计.md)。

另有独立的 20 道算法题、100 个本地用例，排除在这次新增训练划分外。完整的 3,422 题历史题面库已部分用于训练，不能整体再算独立评估。见[数据集说明](datasets/leetcode/数据集说明.md)与[训练数据扩充说明](docs/训练数据扩充说明.md)。

## LeetCode 训练与验证

```powershell
pyforge validate datasets/leetcode/train.jsonl
pyforge validate datasets/leetcode/validation.jsonl
pyforge validate datasets/leetcode/test.jsonl
# 4080 SUPER 完整数据：1.5B、2048 token、2 轮；使用新输出目录
pyforge train --config configs/train_4080.json --data datasets/leetcode --output outputs/train-4080-001 --offline
# 10 步短冒烟：使用短的历史数据
pyforge train --config configs/smoke_4060.json --data datasets/leetcode/archive/codex_v1 --output outputs/smoke-new-001 --offline
```

validate 只检查数据格式；训练启动时额外检查跨集合 ID、题族与重复对话。验证损失自动计算，test 不参与损失或检查点选择。不要用 512 token smoke 配置训练全部新增问答，否则会大量过滤。4080 配置细节与首次运行检查见训练运行指南。

```powershell
Get-Content outputs/train-4080-001/training_log.jsonl -Tail 10 -Wait
Get-Content outputs/train-4080-001/metrics.json
# 中断后使用实际存在的完整检查点，配置/数据/依赖必须与原运行一致
pyforge train --config configs/train_4080.json --data datasets/leetcode --output outputs/train-4080-001 --offline --resume outputs/train-4080-001/checkpoint-100
```

恢复不能从 adapter/ 开始，不能换配置或数据，也不能延长已完成计划。最终 adapter/ 导出验证损失最低的已保存检查点，不一定是最后一步。完整说明见[训练运行指南](docs/训练运行指南.md)。

## 推理与绘图

```powershell
pyforge chat --run outputs/train-4080-001 --offline --max-new-tokens 512 --prompt "编写保持原顺序的列表去重函数，说明输入限制并给出边界条件测试。"
python scripts/plot_training.py outputs/train-4080-001
python scripts/plot_training.py outputs/train-4080-001 --style trend --window 20
```

图片默认保存到运行目录 plots/。趋势图蓝线为 20 点分段均值，红线为原始验证损失，空心菱形为不足一个窗口的均值；不能用它代替代码执行通过率。

## LeetCode 模型的独立算法评估

推荐用一条命令完成“同一 20 题分别发给基础模型与微调模型 → 相同 Docker 测试 → 逐题表格”：

```powershell
pyforge benchmark --run outputs/train-4080-001 --output outputs/benchmark-4080-001
```

上例用于用当前修复后数据新训练的运行。旧 train-doocs-001 绑定修复前的数据哈希，评估前重建旧数据到独立目录（目录已存在则复用），再显式指定 --data：

```powershell
python scripts/repair_training_interfaces.py --restore --output outputs/data-before-interface-repair
pyforge benchmark --run outputs/train-doocs-001 --data outputs/data-before-interface-repair --output outputs/benchmark-001
```

默认离线加载项目模型缓存，生成上限 1024 token，执行超时每题 10 秒。Docker Desktop 与 python:3.12-slim 镜像必须已准备好；先自检，再开始耗时生成。基础模型指未加载本次 LoRA 的原始模型。两次生成在独立进程中顺序执行，避免同时占用 GPU。

命令自动完成五步：基础预测、微调预测、基础执行、微调执行、生成对照。结束时终端打印 20 行结果表，并保存：

- outputs/benchmark-4080-001/comparison/逐题答案对比.html：结果表，可展开同一题面与两边完整原始回答。
- outputs/benchmark-4080-001/comparison/逐题对比.md：20 行结果与汇总。
- outputs/benchmark-4080-001/comparison/comparison.json：机器可读数据与来源哈希。
- 输出根目录的两份预测、生成清单、执行报告与 pipeline_status.json。

输出目录必须是新目录；中断或失败会记录阶段，不会伪造完成状态。当前一键入口不支持从中间阶段恢复；已有完整预测/报告时可用下面的分步入口，不必重新生成。自动生成的是 HTML/Markdown 表格，不依赖 Excel 或桌面应用；此前提供的 xlsx 是该次对照的额外导出文件。

Docker Desktop 需已启动并使用 Linux 容器。预测只读取 prompt，不提供 tests 或参考答案：

```powershell
docker pull python:3.12-slim
python scripts/predict_eval.py --run outputs/train-doocs-001 --data outputs/data-before-interface-repair --output outputs/doocs_finetuned_predictions_v2.jsonl
pyforge evaluate --suite datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl --predictions outputs/doocs_finetuned_predictions_v2.jsonl --report outputs/doocs_finetuned_report_v2.json --timeout 10
Get-Content outputs/doocs_finetuned_report_v2.json
```

基础模型对比保持同样参数，增加 --base 并换文件名：

```powershell
python scripts/predict_eval.py --run outputs/train-doocs-001 --data outputs/data-before-interface-repair --base --output outputs/doocs_base_predictions_v2.jsonl
pyforge evaluate --suite datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl --predictions outputs/doocs_base_predictions_v2.jsonl --report outputs/doocs_base_report_v2.json --timeout 10
```

两边生成和执行完成后，按相同题号创建逐题对照表：

```powershell
python scripts/compare_eval.py --base-predictions outputs/doocs_base_predictions_v2.jsonl --base-report outputs/doocs_base_report_v2.json --finetuned-predictions outputs/doocs_finetuned_predictions_v2.jsonl --finetuned-report outputs/doocs_finetuned_report_v2.json --output outputs/comparison-v2
```

输出逐题对比.md、逐题答案对比.html、comparison.json。表格展示两边状态和改善/退步，HTML 可展开题面及两边完整原始答案。脚本核对同一题库、模型快照、训练清单、生成参数、Docker 镜像及超时，条件不同会拒绝比较。

重点看 passed、total、pass_at_1 和 results。tokens 为生成 token 数，truncated=False 只表示未因长度上限截断，不代表正确。预测与报告拒绝覆盖，重复评估换新文件名。这些命令评估保留的 20 道算法题，不是新增的 123 条问答；本地用例也不是官方隐藏测试。见[算法评估运行指南](docs/算法评估运行指南.md)。

## 已提交的实验记录

[reports/train-doocs-001](reports/train-doocs-001/实验说明.md) 含真实清单、1,146 步日志、指标和两边各 20 条原始预测及执行报告。最佳检查点 checkpoint-1100，验证损失 0.3515698。同条件结果：基础模型 12/20（60%），微调模型 10/20（50%），2 题改善、4 题退步、8 题均通过、6 题均未通过。本轮在这 20 题上净下降 10 个百分点，不应宣称微调收益。

完整表格和答案见[逐题对比](reports/train-doocs-001/comparison/逐题对比.md)、[可展开答案对比](reports/train-doocs-001/comparison/逐题答案对比.html)及同目录的模型评估对比.xlsx；Excel 含逐题结果和完整题面/回答两个工作表。

克隆后可直接绘图或执行已提交的预测，无需模型：

```powershell
python scripts/plot_training.py reports/train-doocs-001 --style trend --output outputs/archived-loss-trend.png
pyforge evaluate --suite datasets/leetcode/benchmarks/doocs-leetcode/executable_v1.jsonl --predictions reports/train-doocs-001/predictions.jsonl --report outputs/archived-evaluation-v1.json --timeout 10
```

导出新实验（有匹配执行报告时增加 --evaluation）：

```powershell
python scripts/export_experiment.py --run outputs/train-doocs-001 --output reports/train-doocs-001-new --predictions outputs/doocs_finetuned_predictions_v1.jsonl
```

轻量归档不含权重或优化器，不能单凭它推理或恢复训练。模型、checkpoint、虚拟环境、上游缓存、历史原始输出和 .env 留在本地并忽略；数据、配置、源码、测试和选定的轻量记录提交 Git。

## 文档导航

- [训练数据审计](docs/训练数据审计.md)：固定 200 题执行审计、接口问题和后续对照实验；[本次报告](reports/data-audit-v1/审计说明.md)。
- [项目文件与归档说明](docs/项目文件与归档说明.md)：目录、脚本依赖、清理与提交范围。
- [开发文档](docs/开发文档.md)：当前模块、契约、检查策略和后续工作。
- [训练运行指南](docs/训练运行指南.md)、[本地开发指南](docs/本地开发指南.md)：安装、训练、恢复与日志。
- [数据与评估规范](docs/数据与评估规范.md)、[训练数据扩充说明](docs/训练数据扩充说明.md)：数据及评估边界。
- [算法评估运行指南](docs/算法评估运行指南.md)、[算法验证集说明](docs/算法验证集说明.md)：独立执行题和历史来源目录。
- [训练验证记录](docs/训练验证记录.md)、[旧留出集测试报告](docs/留出集测试报告.md)：历史实测证据。
- [题库说明](docs/题库说明.md)、[答案编写与检查指南](docs/答案编写与检查指南.md)：原始 100 题与重建。
- [历史教师数据生成指南](docs/教师数据生成指南.md)：旧硅基流动流程，不是当前默认工作流。

基础模型来源：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)；题解来源：[doocs/leetcode](https://github.com/doocs/leetcode)，许可副本随数据保留。项目代码尚未选定许可证，公开仓库不代表自动获得任意使用授权。
