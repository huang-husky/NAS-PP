"""评估协议的实现（见 PROTOCOL.md）：功能等价去重 + 按等价类切分 + 指标。

之后所有实验（基线、微调）都从这里取数据和切分，保证大家比的是同一批结构。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "predictor_lab"))
from nb201 import DATA_FILE, EDGES, OPS, load_dataset  # noqa: E402

N_TEST_CLASSES = 2000
N_GRID = [20, 50, 100, 200, 500, 1000, 2000]


# ======================================================================================
# 第 1 段：功能等价（规范形式）
# ======================================================================================
def canonical(row):
    """把"不在任何 输入->输出 通路上的边"的操作换成 none，得到规范形式。
    row: 6 个操作编号（顺序同 EDGES，0 = none）。"""
    live = [(u, v) for (u, v), o in zip(EDGES, row) if o != 0]
    reach = {0}                       # 从节点 0 能走到的节点
    changed = True
    while changed:
        changed = False
        for u, v in live:
            if u in reach and v not in reach:
                reach.add(v)
                changed = True
    back = {3}                        # 能走到节点 3 的节点
    changed = True
    while changed:
        changed = False
        for u, v in live:
            if v in back and u not in back:
                back.add(u)
                changed = True
    return tuple(o if (o != 0 and u in reach and v in back) else 0 for (u, v), o in zip(EDGES, row))


class Data:
    """15625 个结构的全部信息 + 等价类。"""

    def __init__(self):
        archs, acc = load_dataset()                         # 标签：cifar10_acc
        self.ops = np.array([[OPS.index(o) for o in a] for a in archs])      # (15625, 6)
        self.y = acc
        df = pd.read_csv(DATA_FILE)
        self.flops = df["flops_M"].values.astype(float)
        self.params = df["params_M"].values.astype(float)
        self.canon = np.array([canonical(r) for r in self.ops])              # (15625, 6)
        keys = pd.Series([tuple(r) for r in self.canon])
        self.gid = keys.astype("category").cat.codes.values                  # 每个结构的等价类编号
        self.n_groups = int(self.gid.max()) + 1
        self.members = [np.where(self.gid == g)[0] for g in range(self.n_groups)]

    # ---------------- 特征 ----------------
    @staticmethod
    def onehot(a):
        return np.concatenate([(a[:, [i]] == k).astype(float) for i in range(6) for k in range(5)], axis=1)

    @staticmethod
    def counts(a):
        return np.stack([(a == k).sum(1) for k in range(5)], axis=1).astype(float)

    def feature_sets(self):
        fp = np.stack([np.log1p(self.flops), np.log1p(self.params)], axis=1)
        oh, ohc = self.onehot(self.ops), self.onehot(self.canon)
        return {
            "onehot": oh,
            "onehot+fp": np.hstack([oh, fp]),
            "canon": ohc,
            "canon+fp": np.hstack([ohc, fp]),
            "counts+onehot": np.hstack([self.counts(self.ops), oh]),
        }

    # ---------------- 切分 ----------------
    def split(self, seed, n):
        """返回 (训练结构下标, 测试结构下标)。
        测试类固定为该种子排列的最后 N_TEST_CLASSES 个类，不随 n 变化；训练类取前 n 个（嵌套）。"""
        order = np.random.default_rng(seed).permutation(self.n_groups)
        rep_rng = np.random.default_rng(seed + 10_000)
        rep = np.array([m[rep_rng.integers(len(m))] for m in self.members])   # 每个类的代表结构
        train = rep[order[:n]]
        test = rep[order[-N_TEST_CLASSES:]]
        return train, test


# ======================================================================================
# 第 2 段：指标
# ======================================================================================
def evaluate(pred, y_true):
    """返回 Kendall tau-b、Spearman，以及 regret@10（越小越好）。"""
    top10 = np.argsort(-pred)[:10]
    return dict(tau=kendalltau(pred, y_true)[0], rho=spearmanr(pred, y_true)[0],
                regret10=float(y_true.max() - y_true[top10].max()))
