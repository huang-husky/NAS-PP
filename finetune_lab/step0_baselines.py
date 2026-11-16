"""第 0 步：在新协议（按功能等价类切分）下，重算所有对照基线的学习曲线。

做三件事
    1. 自检：切分里训练和测试没有功能等价的结构，测试集不随 n 改变，训练集嵌套。
    2. 简单基线：随机森林配各种"免费"特征（one-hot、规范形式、FLOPs 与参数量、计数）；梯度提升。
    3. 冻结 LLM 对照：沿用 llm_demo/features/ 里已提取的 Pythia-410M 特征，随机森林，同一套切分。
最后算出"包络"（每个 n 下最强的简单基线），它就是后面微调方案要超过的对手。

用法
    python -I step0_baselines.py                 全部（约 10 分钟）
    python -I step0_baselines.py --quick         冒烟测试：2 个种子、3 个 n
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from protocol import N_GRID, Data, evaluate  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
FEATURE_DIR = HERE.parent / "llm_demo" / "features"
LLM_FEATURES = {"frozen_pythia_random": "pythia410m_step0.npy", "frozen_pythia_trained": "pythia410m_step143000.npy"}


def self_check(d):
    """切分的三条性质（见 PROTOCOL.md 第 3 节）。"""
    for seed in (0, 1, 2):
        tests = [d.split(seed, n)[1] for n in N_GRID]
        assert all((t == tests[0]).all() for t in tests), "测试集随 n 变了"
        te_groups = set(d.gid[tests[0]])
        prev = None
        for n in N_GRID:
            tr = d.split(seed, n)[0]
            assert len(set(d.gid[tr])) == n, "训练结构里有功能等价的重复"
            assert not (set(d.gid[tr]) & te_groups), "训练和测试有功能等价的结构（泄漏）"
            if prev is not None:
                assert set(prev) <= set(tr), "训练集没有嵌套"
            prev = tr
    print("自检通过：训练/测试没有功能等价结构；测试集固定；训练集嵌套", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default=None, help="逗号分隔的种子，例如 100,101,102（开发种子）；不写则用评估种子 0～9")
    ap.add_argument("--grid", default=None, help="逗号分隔的 n，例如 100,200,500；不写则用协议里的全部 n")
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else ([0, 1] if args.quick else list(range(10)))
    grid = [int(s) for s in args.grid.split(",")] if args.grid else ([50, 200, 1000] if args.quick else N_GRID)
    llm_seeds = seeds[:5]                      # 冻结 LLM 的 1024 维特征慢，只用前 5 个种子

    d = Data()
    print(f"{len(d.y)} 个结构 -> {d.n_groups} 个功能等价类", flush=True)
    self_check(d)
    feats = d.feature_sets()
    llm = {k: np.load(FEATURE_DIR / f) for k, f in LLM_FEATURES.items()}
    for k, a in llm.items():
        assert a.shape[0] == len(d.y), f"{k} 行数和数据不一致"

    rf = lambda X, tr, te, s, trees=200: RandomForestRegressor(trees, n_jobs=-1, random_state=s).fit(X[tr], d.y[tr]).predict(X[te])
    hgb = lambda X, tr, te, s: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=3,
                                                              random_state=s).fit(X[tr], d.y[tr]).predict(X[te])
    rows, t0 = [], time.time()
    for si, seed in enumerate(seeds):
        for n in grid:
            tr, te = d.split(seed, n)
            preds = {f"rf_{k}": rf(X, tr, te, seed) for k, X in feats.items()}
            preds["hgb_canon+fp"] = hgb(feats["canon+fp"], tr, te, seed)
            if seed in llm_seeds:
                for k, X in llm.items():
                    preds[k] = rf(X, tr, te, seed, trees=100)
            for name, p in preds.items():
                rows.append(dict(seed=seed, n=n, method=name, **evaluate(p, d.y[te])))
        el = time.time() - t0
        bar = "#" * int(20 * (si + 1) / len(seeds)) + "-" * (20 - int(20 * (si + 1) / len(seeds)))
        print(f"[{bar}] 种子 {seed} 完成 ({si + 1}/{len(seeds)})，已用 {el / 60:.1f} 分钟，"
              f"预计剩余 {el / (si + 1) * (len(seeds) - si - 1) / 60:.1f} 分钟", flush=True)

    res = pd.DataFrame(rows)
    tag = ("_quick" if args.quick else "") + args.tag
    res.to_csv(OUT / f"step0_raw{tag}.csv", index=False)
    tab = res.groupby(["method", "n"]).tau.mean().unstack("n")
    simple = [m for m in tab.index if not m.startswith("frozen")]
    env = tab.loc[simple].max(0)
    env_who = tab.loc[simple].idxmax(0)
    print("\n===== Kendall tau-b（按功能等价类切分；测试 = 2000 个互不等价的结构）=====")
    print(tab.round(3).to_string())
    print("\n包络（每个 n 下最强的简单基线）：")
    for n in tab.columns:
        print(f"  n={n:5d}: {env[n]:.3f}  ({env_who[n]})")
    out = tab.copy()
    out.loc["ENVELOPE"] = env
    out.round(4).to_csv(OUT / f"step0_summary{tag}.csv")
    reg = res.groupby(["method", "n"]).regret10.mean().unstack("n")
    print("\nregret@10（预测最好的 10 个里，真实最高精度比测试集最高精度低多少，越小越好）：")
    print(reg.round(2).to_string())


if __name__ == "__main__":
    main()
