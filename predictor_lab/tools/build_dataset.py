"""从 Syne Tune 的表格化 NAS-Bench-201 生成 data/nasbench201.csv（已经跑过，一般不需要再跑）。

原始数据: https://huggingface.co/datasets/synetune/blackbox-repository/tree/main/nasbench201
需要其中两个文件:
    hyperparameters.parquet       15625 个结构，每行 6 个操作名
    objectives_evaluations.npy    约 788MB，形状 (3 个数据集, 15625 个结构, 3 个随机种子, 200 个 epoch, 7 个指标)

用法:
    python tools/build_dataset.py <存放上面两个文件的文件夹>
"""

import argparse
import json
import hashlib
import math
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
    不能仅凭最高精度相近认定验证/测试语义；本导出保留源字段metric_valid_error，轴顺序需用源元数据核对。
    """
    err = evals[TASKS.index(task), :, :, epoch - 1, OBJECTIVES.index("metric_valid_error")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 100.0 * (1.0 - np.nanmean(err, axis=1))  # 没训练过的种子是 NaN


def main(src_dir):
    src_dir = Path(src_dir)
    hps = pd.read_parquet(src_dir / "hyperparameters.parquet")
    evals = np.load(src_dir / "objectives_evaluations.npy", mmap_mode="r")
    if evals.shape != (3, 15625, 3, 200, 7):
        raise ValueError(f"Unexpected objective tensor: {evals.shape}")
    if len(hps) != 15625 or hps[[f"hp_x{i}" for i in range(6)]].duplicated().any():
        raise ValueError("Unexpected or duplicate architecture rows")

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

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        raise FileExistsError(f"Preserve existing data: choose a new --out file, not {OUT_FILE}")
    temporary = OUT_FILE.with_suffix(OUT_FILE.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + [f"op_{s}to{d}" for s, d in EDGES] + list(columns))
        for i in range(len(ops)):
            writer.writerow([i] + list(ops[i]) + [float(col[i]) for col in columns.values()])
    if len(ops) != 15625 or any(not np.isfinite(v).all() for v in columns.values()):
        raise ValueError("Invalid NB201 labels or row count")
    temporary.replace(OUT_FILE)
    print(f"写入 {OUT_FILE}，共 {len(ops)} 行")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()
    OUT_FILE = args.out
    main(args.source)
    metadata = {"source": str(Path(args.source).resolve()), "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(OUT_FILE.read_bytes()).hexdigest(), "row_alignment": "use architecture IDs, never reuse old ordinal pools after regeneration"}
    OUT_FILE.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
