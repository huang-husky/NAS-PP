# 性能预测器入门实验

在 NAS-Bench-201 上亲手做一个最小的性能预测器，对应 survey 第 III-B-1 节（第 8–10 页）。
全部在 CPU 上运行，每个脚本几秒钟跑完。

## 怎么运行

在 `predictor_lab` 文件夹下打开终端：

```bash
python step1_explore.py
```

```bash
python step2_predictor.py
```

## 文件说明

| 文件 | 作用 | 要不要动 |
|---|---|---|
| `step1_explore.py` | 认识数据：结构长什么样、精度怎么分布 | 直接运行，读懂每一行，想末尾的两道思考题 |
| `step2_predictor.py` | 预测器骨架：编码、划分、训练、评价 | **补全 4 个 TODO**，每补一个运行一次 |
| `step3_mlp_ranking.py` | PyTorch 神经网络预测器，比较 MSE 损失和排序损失（约 20 秒） | 直接运行 |
| `checks.py` | 每个 TODO 的自动检查，没通过会提示原因 | 不用改 |
| `nb201.py` | 读取数据的工具函数 | 不用改，但值得读一遍 |
| `data/nasbench201.csv` | 15625 个结构和它们的精度 | 可以用 Excel 打开看看 |
| `tools/build_dataset.py` | 生成上面这个 CSV 的脚本，记录了数据来源 | 不用管 |
| `outputs/` | 脚本画的图 | 自动生成 |

## 数据

`data/nasbench201.csv` 每行是一个结构：

- `op_0to1` … `op_2to3`：cell 里 6 条边各自选的操作
- `cifar10_acc`、`cifar100_acc`、`imagenet16_acc`：训练 200 轮后的精度（百分数），对多个随机种子取平均
- `cifar10_acc_epoch12`：CIFAR-10 上只训练到第 12 轮时的精度
- `flops_M`、`params_M`：计算量和参数量（百万）

数据来自 Syne Tune 整理的表格化 NAS-Bench-201（原文件 788MB，已校验 SHA-256），这里只抽取了需要的列。
CIFAR-10 的最高精度是 94.37（第 6111 号结构），与 NAS-Bench-201 论文报告的最优结构和精度一致。

## 常见情况

- 运行时出现 `UserWarning: Pandas requires version ... of 'numexpr'`：这是 Anaconda 里两个包版本不匹配的提醒，不影响结果，忽略即可。
- 出现 `[TODO x 未通过]`：正常，按提示改完再运行。
- 看不懂报错：把完整的报错信息贴给 Claude。
