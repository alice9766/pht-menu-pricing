# 实验代码与完整证明上传版

本包仅用于发布实验代码和完整证明。配置、随机种子、依赖文件和运行说明作为代码的必要组成部分一并保留。

上传范围包括：

- `proofs/`：完整证明 PDF、TeX 及编号对照。
- `experiments/`：精确算法、比较方法、数据构造、实验运行、结果分析与测试代码，以及固定实验配置。
- `scripts/`：生成正文实验图表的代码和菜单示例提取代码。
- `reproduce.py`、`requirements.txt`、两个 README、`.gitignore`、`.gitattributes`。

包内没有论文正文、原始时序数据、构造后的实例、历史实验结果、日志或已经生成的图片。`experiments/source/legacy_cost_reference.json` 是确定性成本配置及生成规则，不是实验结果。

## 如何运行

推荐 Linux 或 WSL2，使用 Python 3.12.14 与 `requirements.txt` 中的依赖。完整命令见英文 README。主要顺序为：

```bash
python -m pip install -r requirements.txt
python reproduce.py tests
python reproduce.py raw-data
python reproduce.py inputs
python reproduce.py run --mode bulk --workers 4
python reproduce.py run --mode isolated --workers 4
python reproduce.py summarize
python reproduce.py figures
```

`raw-data` 从 UCI 官方地址下载数据并校验；`inputs` 按固定配置构造输入；两个 `run` 命令执行既定任务；`summarize` 和 `figures` 从运行记录生成统计结果与实验图表。完整证明可直接阅读，也可用 `python reproduce.py proofs` 编译。

运行产生的数据和结果只写入 `.work/` 与 `generated/`，默认不会被 Git 提交。未随包发布历史结果，因此读者需要重新运行实验后生成图表；耗时及限时算法结果可能随机器环境变化。

## 如何上传到 GitHub

1. 创建一个新的空仓库，例如 `pht-code-and-proofs`。
2. 解压 `PHT_Code_and_Proofs.zip`，把解压目录中的内容放在仓库根目录，让 README 能直接显示。
3. 上传上面列出的文件夹和文件，保留 `.gitignore` 与 `.gitattributes`。不要添加本地运行产生的 `.work/`、`generated/` 或虚拟环境。
4. 提交后，在网页中确认能打开 `README.md`、`reproduce.py` 和 `proofs/complete_proofs.pdf`。
5. 将仓库链接发回，届时再更新论文中的配套材料引用与发布时态。

若通过 GitHub Desktop 操作，先把空仓库克隆到电脑，再复制本包内容，检查待提交文件列表，随后提交并推送即可。本次不上传此前包含正文、数据和结果的完整复现包。
