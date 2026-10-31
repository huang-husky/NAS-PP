"""NAS-Bench-201 的读取工具（已写好，直接用，不需要改）。

NAS-Bench-201 的搜索空间：每个网络由同一个 cell 堆叠而成，cell 是一个有 4 个节点
（编号 0、1、2、3）的有向无环图。节点之间共有 6 条边，每条边要从 5 种操作里选 1 种，
所以一共有 5^6 = 15625 个不同的结构。
"""

import csv
from pathlib import Path

import numpy as np

# 每条边可选的 5 种操作
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

# 6 条边，(起点, 终点)。一个结构里 6 个操作的顺序与这里一一对应
EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

DATA_FILE = Path(__file__).parent / "data" / "nasbench201.csv"

# 图片统一保存到这个文件夹
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_OP_COLUMNS = [f"op_{src}to{dst}" for src, dst in EDGES]


def load_dataset(label="cifar10_acc"):
    """读取全部 15625 个结构和它们的精度。

    返回:
        archs: 长度 15625 的列表，每个元素是一个结构，用 6 个操作名表示，例如
               ['avg_pool_3x3', 'nor_conv_1x1', 'skip_connect', ...]
        acc:   长度 15625 的 numpy 数组，acc[i] 是 archs[i] 在 CIFAR-10 上训练 200 轮之后的精度（百分数）

    label 可以换成 CSV 里的其他列，例如 'cifar100_acc'。
    """
    archs, acc = [], []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            archs.append([row[col] for col in _OP_COLUMNS])
            acc.append(float(row[label]))
    return archs, np.array(acc)


def to_arch_str(arch):
    """把 6 个操作名拼成 NAS-Bench-201 论文里的标准写法，读论文时会见到。"""
    a = arch
    return (
        f"|{a[0]}~0|+"
        f"|{a[1]}~0|{a[2]}~1|+"
        f"|{a[3]}~0|{a[4]}~1|{a[5]}~2|"
    )
