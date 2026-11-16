# NB101 与 NB201 的少样本迁移

给预测器一个做过的大题库，再给它一个新题库的少量答案。比较 Pythia-70M 和图神经网络谁更能利用旧题库，提高新题库里的架构排序。

在 VS Code 里打开仓库根目录，在终端依次运行需要的阶段。

```powershell
python transfer_lab/run_transfer.py --preset smoke
python transfer_lab/run_transfer.py --preset pilot
python transfer_lab/run_transfer.py --preset full
```

| 阶段 | 源空间样本 | 目标标签 | 目标测试样本 | 种子 | 每次源／目标训练步数 |
|---|---:|---|---:|---|---|
| smoke | 32 | 10 | 24 | 100 | 2／2 |
| pilot | 1000 | 50 | 2000 | 100–102 | 300／150 |
| full | 1000 | 10、50、100 | 2000 | 0–4 | 300／150 |

先跑 smoke 检查环境，再跑 pilot 看方向。CPU 上 Pythia 会花较长时间，终端会显示方向、种子和训练步。已有 CUDA 版 PyTorch 时可加 `--device cuda`。只想检查 GNN 分支可以加 `--gnn-only`。默认使用本地 `llm_demo/models/pythia70m_step143000/`，也可通过 `--model-path` 指向其他本地目录。脚本读取 `predictor_lab/data/` 下的两个 NASBench CSV，运行依赖见根目录的 `requirements-audited.txt`。

## 对照

- 双向运行 NB101 → NB201 和 NB201 → NB101。NB101 使用本地 CSV 的 CIFAR-10 `test_acc`，NB201 使用 `cifar10_acc`。NB101 以数据文件中的去重架构为抽样单位，NB201 复用微调实验的功能等价类切分。每个方向和种子内共用目标测试集，目标训练集随标签预算嵌套。
- 两个预测器输入同一份带操作类型的有向图。Pythia 读节点和边的文字列表并全量微调，GNN 对图做双向消息传递，并使用节点类型计数。这是保留连接和操作的简化 cell 表示，宏观网络配置及 NB101 的投影细节省略。
- 对每个模型比较 `target_only` 和 `source_then_target`。两组共用起始权重、目标样本、目标批次和目标训练步数。迁移组多经过源空间训练，并用新的优化器学习目标空间。每个训练批次只比较一个空间内部的架构好坏，精度完全相同的架构对跳过。
- 内置 GNN 是本轮对照。若有正向信号，后续再用 DAP 等已发表的跨空间方法和新的留出空间验证。

## 看结果

默认输出到 `transfer_lab/results/<阶段>/`。同一命令重跑会跳过已完成的配置，若源阶段中断则会从该次源训练重来。更改设备、学习率或脚本版本后，请用 `--out` 指定新的结果目录。每个结果目录供一个运行进程使用。

- `runs.csv` 记录每次运行的 Kendall τ-b、Spearman ρ、`regret10` 和训练时间。
- `summary.csv` 记录各组的均值、标准差和有效样本数。
- `paired.csv` 记录每个种子的迁移增益、迁移后 Pythia 与 GNN 的差值，以及两者迁移增益之差。τ 和 ρ 的正差值表示前者更高，`regret10` 的负差值表示前者更好。Regret 的单位为精度百分点，计算测试集中最好架构与预测前十中最好架构的真实精度差。

Pilot 先看 `tau_pythia_transfer_gain`、`tau_pythia_minus_gnn_after_transfer` 和对应的 regret 差值。两个方向都出现一致信号，再投入正式评估。Smoke 只用于确认训练、预测和写表能运行。

## DeepSeek 重排序小试验

每轮从 NB201 抽 50 个已标注架构，用原协议在 50 标签时表现最强的结构随机森林（规范化操作 one-hot、FLOPs、参数量）将 2,000 个留出架构筛到 20 个。随机森林直接选自己的前 5 个。DeepSeek 看同样的 50 个标注例子和打乱后的 20 个候选，每个候选附带结构、FLOPs 和参数量，再选 5 个。提示词里的候选 ID 每轮重新分配。两种选法都用后验真实最高精度计分，因此每种选法的预算是 50 个已有标签和 5 次候选评估。

```powershell
python transfer_lab/run_rerank.py --dry-run
python transfer_lab/run_rerank.py --smoke
python transfer_lab/run_rerank.py --run
```

- `--dry-run` 只在本地跑 20 个种子，导出随机森林结果和一份提示词。候选真实精度、数据集原始结构 ID 和随机森林分数只留在本地。
- `--smoke` 用单独的开发种子发一条请求。确认输出正常后，用 `--run` 跑 20 个评估种子，最多需要 20 条成功请求。有效回复会按请求内容缓存，中断后可用同一条命令继续。在线模式沿用终端里的 `DEEPSEEK_API_KEY`，默认模型是 `deepseek-v4-pro`，关闭思考。
- 输出位于 `transfer_lab/results/rerank/deepseek-v4-pro/`。各阶段的 `baseline.csv` 记录随机森林成绩和候选池能达到的最高成绩。在线阶段的 `paired.csv` 记录每轮选出的最好成绩和差值，`summary.json` 汇总平均差值及胜、平、负次数。
- 事先固定试验门槛为 DeepSeek 平均提升至少 **0.05 个精度百分点**，同时赢的轮数多于输的轮数。各轮的候选架构可能重叠。达标后需要在独立搜索空间复验。
