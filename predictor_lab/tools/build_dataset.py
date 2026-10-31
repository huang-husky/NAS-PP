"""从 Syne Tune 的表格化 NAS-Bench-201 生成 data/nasbench201.csv（已经跑过，一般不需要再跑）。

原始数据: https://huggingface.co/datasets/synetune/blackbox-repository/tree/main/nasbench201
需要其中两个文件:
    hyperparameters.parquet       15625 个结构，每行 6 个操作名
    objectives_evaluations.npy    约 788MB，形状 (3 个数据集, 15625 个结构, 3 个随机种子, 200 个 epoch, 7 个指标)

用法:
    python tools/build_dataset.py <存放上面两个文件的文件夹>
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["cifar10", "cifar100", "ImageNet16-120"]
OBJECTIVES = [
    "metric_valid_error", "metric_train_error", "metric_runtime",
    "metric_elapsed_time", "metric_latency", "metric_flops", "metric_params",
]
EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
OUT_FILE = Path(__file__).parent.parent / "data" / "nasbench201.csv"


def accuracy(evals, task, epoch):
    """某个数据集在第 epoch 轮的精度（百分数），对训练过的随机种子取平均。

    Syne Tune 把这个指标叫 metric_valid_error。CIFAR-10 上的最大值是 94.37（第 6111 号结构），
    与 NAS-Bench-201 论文里 CIFAR-10 的最优测试精度一致，说明它对应的是用完整训练集训练的那一组结果。
    """
    err = evals[TASKS.index(task), :, :, epoch - 1, OBJECTIVES.index("metric_valid_error")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 100.0 * (1.0 - np.nanmean(err, axis=1))  # 没训练过的种子是 NaN


def main(src_dir):
    src_dir = Path(src_dir)
    hps = pd.read_parquet(src_dir / "hyperparameters.parquet")
    evals = np.load(src_dir / "objectives_evaluations.npy", mmap_mode="r")
    assert evals.shape == (3, 15625, 3, 200, 7), evals.shape

    ops = hps[[f"hp_x{i}" for i in range(6)]].to_numpy()
    c10 = TASKS.index("cifar10")
    columns = {
        "cifar10_acc": accuracy(evals, "cifar10", 200),
        "cifar100_acc": accuracy(evals, "cifar100", 200),
        "imagenet16_acc": accuracy(evals, "ImageNet16-120", 200),
        "cifar10_acc_epoch12": accuracy(evals, "cifar10", 12),
        "flops_M": np.nanmean(evals[c10, :, :, 0, OBJECTIVES.index("metric_flops")], axis=1),
        "params_M": np.nanmean(evals[c10, :, :, 0, OBJECTIVES.index("metric_params")], axis=1),
    }

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + [f"op_{s}to{d}" for s, d in EDGES] + list(columns))
        for i in range(len(ops)):
            writer.writerow([i] + list(ops[i]) + [f"{col[i]:.4f}" for col in columns.values()])
    print(f"写入 {OUT_FILE}，共 {len(ops)} 行")


if __name__ == "__main__":
    main(sys.argv[1])
