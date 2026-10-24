# PyForge

面向个人 NVIDIA GPU 的中文需求转 JSON 数据清洗配置模型，使用 Python 3.12、Transformers Trainer、PEFT 和 bitsandbytes 进行 SFT + QLoRA。当前工作目录为 `E:\code\github\pyforge`，正式训练与推理使用 RTX 4080 SUPER（16 GB）和 Qwen2.5-Coder-1.5B-Instruct。

模型负责生成去重、空值处理、筛选和排序配置，由固定本地执行器处理数据。训练、测试、验证、Excel 对比和使用示例已在本地跑通；本地权重与运行输出不提交 Git。

## 环境准备

准备 Python 3.12 虚拟环境并激活后安装：

```powershell
Set-Location E:\code\github\pyforge
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train,plot]"
pyforge doctor
```

首次运行需提前下载基础模型；已缓存后使用离线命令：

```powershell
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --cache-dir E:/code/github/pyforge/models/hub
```

项目不需要 Docker 或模型 API 密钥。Excel 导出使用本机 Codex 附带的 Node.js/artifact-tool。完整安装说明见[本地开发指南](docs/本地开发指南.md)。

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

## 当前数据

| 集合 | 条数 | 用途 |
| --- | ---: | --- |
| datasets/data_cleaning/train.jsonl | 1,000 | 更新模型参数 |
| datasets/data_cleaning/test.jsonl | 200 | 测试预测与配置匹配 |
| datasets/data_cleaning/validation.jsonl | 200 | 验证 loss 和开发评估 |

需求与答案由 Codex 和经授权的三个子代理逐条编写，代码只封装、抽样和检查。完整作者原稿保留在 sources/，检查记录在 metadata/，缩小前的完整版本在 archive/full_v1/。默认训练与评估只使用 datasets/data_cleaning，历史硅基流动数据只在 datasets/archive/teacher_v1/ 保留，不参与当前流程。见[数据集说明](datasets/data_cleaning/数据集说明.md)。

## 检查与文档

```powershell
python -m unittest discover -s tests -v
python scripts/check_data_cleaning_cases.py
```

- [数据清洗数据集指南](docs/数据清洗数据集指南.md)：训练、测试、验证、一键 Excel 对比和使用示例。
- [训练运行指南](docs/训练运行指南.md)：日志、检查点、恢复与模型文件。
- [本地开发指南](docs/本地开发指南.md)：环境安装与检查。
- [数据与评估规范](docs/数据与评估规范.md)：来源、划分、严格与诊断指标。
- [开发文档](docs/开发文档.md)：当前模块和接口。
- [项目文件与归档说明](docs/项目文件与归档说明.md)：目录、保留范围与备份。

基础模型来源：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)。项目代码和自编数据尚未选定开放许可证。
