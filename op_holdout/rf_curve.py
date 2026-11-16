"""随机森林的"标签数 -> 排名质量"曲线（在和 DeepSeek 完全相同的测试架构上）。

目的：把"DeepSeek 用 30 个标签达到 0.60"换算成"随机森林需要多少标签才追得上"。
不联网，几十秒。训练标签只来自不含被留出操作的架构（没见过的设定）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_op_holdout import DATA, EDGES, HAND, OPS, TARGET, arch_features, fit_predict  # noqa: E402
from run_deepseek import N_TEST  # noqa: E402

df = pd.read_csv(DATA).dropna(subset=[TARGET]).reset_index(drop=True)
op_idx = df[EDGES].apply(lambda s: s.map(OPS.index)).values
y = df[TARGET].values
feats = {"onehot": arch_features(op_idx, np.eye(5)),
         "handcrafted": arch_features(op_idx, np.array([HAND[o] for o in OPS], dtype=float))}

NS = [10, 30, 50, 100, 200, 500, 1000]
rows = []
for held in OPS:
    h = OPS.index(held)
    has_h = (op_idx == h).any(1)
    pool = np.where(~has_h)[0]
    test = np.random.default_rng(100 + h).choice(np.where(has_h)[0], N_TEST, replace=False)  # 与 run_deepseek 相同
    for n in NS:
        for seed in range(5):
            tr = np.random.default_rng(seed).choice(pool, n, replace=False)
            for name, X in feats.items():
                p = fit_predict("rf", X[tr], y[tr], X[test], seed)
                rows.append(dict(held=held, n=n, enc=name, rho=spearmanr(p, y[test])[0]))
res = pd.DataFrame(rows)
tab = res.groupby(["enc", "n"]).rho.mean().unstack("n")
print("随机森林（没见过被留出操作）：训练标签数 -> 平均 Spearman（5 个操作 x 5 个种子）")
print(tab.round(3).to_string())
sd = res.groupby(["enc", "n", "held"]).rho.mean().groupby(["enc", "n"]).std().unstack("n")
print("\n各操作之间的标准差（衡量不同操作间的波动）")
print(sd.round(3).to_string())
res.to_csv(HERE / "results" / "rf_curve_raw.csv", index=False)
