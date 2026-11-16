"""第 3 步（解释实验）：LLM 特征是不是主要记住了"每种操作各用了几次"？

背景：全空间实验发现，LLM 特征小样本时比 one-hot 好，样本多了却卡在 τ≈0.64 左右。
先导实验的猜想：LLM 特征主要编码了"操作计数"——5 种操作各出现在几条边上（5 个数），
而不太在意它们各在哪条边上。这个脚本从三个方面检验这个猜想:

1. 学习曲线：把"操作计数"（5 维）和"计数 + one-hot"（35 维）也当成特征来源，
   和 LLM 特征、one-hot 放在一起比，三个数据集 × 三种回归头 × 全部样本数 × 20 个种子
2. 计数的组均值参考：如果只知道操作计数，最好能排得多准？（同一种计数组合里的结构，
   只能给出同一个预测值，取它们真实精度的平均值，这是使用全体标签的描述性参考，不是排序上限）
3. 线性探针：LLM 特征里能读出多少"计数"信息、多少"每条边放了什么"的信息？
   以及特征的变化有多少能被计数解释、多少能被完整的 one-hot 解释

CIFAR-10 上 one-hot 和 LLM 特征的结果在 evaluate.py 里已经算过，这里直接复用，不重复计算
（训练集的抽法完全一样：同一个种子、同一个结构池）。

运行方式（在 llm_demo 文件夹下）:
    python explain.py                先用默认设置跑完整实验
    python explain.py --seeds 2      冒烟测试：2 个种子快速跑一遍，确认没有报错
    python explain.py --jobs 4       同时用几个进程（默认 8；内存紧张就调小）
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, wilcoxon
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arch_text import load_dataset, one_hot
from evaluate import HEADS, TRAIN_SIZES, make_head, n_at_k

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "features"
OUT_DIR = HERE / "results" / "step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["cifar10_acc", "cifar100_acc", "imagenet16_acc"]
# 换数据集时只带 3 个代表性 LLM：随机初始化、训练完成的 Pythia，以及 Qwen（省时间）
LLMS_ALL = ["pythia410m_step0", "pythia410m_step1000", "pythia410m_step10000",
            "pythia410m_step143000", "qwen25_0.5b", "qwen25coder_0.5b"]
LLMS_MAIN = ["pythia410m_step0", "pythia410m_step143000", "qwen25_0.5b"]
HAND = ["onehot", "counts", "counts+onehot"]

# 画图时每种特征的样式：手工特征用粗线
STYLE = {
    "onehot": dict(color="black", linewidth=2.5, label="one-hot"),
    "counts": dict(color="tab:green", linewidth=2.5, linestyle="--", label="op counts (5-d)"),
    "counts+onehot": dict(color="tab:purple", linewidth=2, label="counts + one-hot"),
    "pythia410m_step0": dict(color="tab:blue", label="Pythia step 0 (random)"),
    "pythia410m_step143000": dict(color="tab:red", label="Pythia step 143000"),
    "qwen25_0.5b": dict(color="tab:orange", label="Qwen2.5-0.5B"),
}


# ---------------------------------------------------------------- 特征

def op_counts(onehot):
    """把 30 维 one-hot（6 条边 × 5 种操作）按边加起来，得到 5 种操作各出现几次。"""
    return onehot.reshape(-1, 6, 5).sum(axis=1)


def build_features(archs):
    from artifacts import verify_snapshot
    verify_snapshot(FEATURE_DIR)
    if not np.array_equal(np.load(FEATURE_DIR / "pool.npy"), np.arange(len(archs))):
        raise ValueError("explain requires the full ordered NB201 pool")
    oh = np.array([one_hot(a) for a in archs])
    feats = {"onehot": oh, "counts": op_counts(oh), "counts+onehot": np.hstack([op_counts(oh), oh])}
    for name in LLMS_ALL:
        feats[name] = np.load(FEATURE_DIR / f"{name}.npy")
    return feats


# ---------------------------------------------------------------- 1. 学习曲线

def run_one(X, y, n, seed, head):
    """Compatibility wrapper. Curve writers use evaluate_once to retain diagnostics."""
    from experiment import evaluate_once
    result = evaluate_once(X, y, n, seed, head)
    return result["tau"], result["n_at_10"]


def learning_curves(feats, n_seeds, n_jobs, sizes):
    from experiment import run_curves, atomic_csv
    parts = []
    for target in TARGETS:
        names = HAND + (LLMS_ALL if target == "cifar10_acc" else LLMS_MAIN)
        parts.append(run_curves("nb201", {target: load_dataset(target)[1]}, names, feats.__getitem__,
                                sizes, n_seeds, n_jobs, OUT_DIR, HEADS, lambda n, k: k))
    df = pd.concat(parts, ignore_index=True)
    atomic_csv(df, OUT_DIR / "curves.csv")
    # run_curves 每调用一次都会写同名的中间文件和质量表；三个数据集分三次调用，这里用合并后的结果重写
    from experiment import write_tau_summary
    atomic_csv(df, OUT_DIR / "curves_nb201_partial.csv")
    write_tau_summary(df, OUT_DIR / "quality_nb201.csv")
    return df


# ---------------------------------------------------------------- 2. 计数的组均值参考

def group_mean_reference(counts, y):
    """分组均值 oracle 参考（非 tau 上界、非留出集泛化分数）：同一种计数组合的结构，都预测为这组结构的真实平均精度。"""
    from experiment import validate_xy
    validate_xy(np.asarray(counts), np.asarray(y))
    _, group = np.unique(counts, axis=0, return_inverse=True)
    group = group.ravel()
    group_mean = np.bincount(group, weights=y) / np.bincount(group)
    return kendalltau(group_mean[group], y).statistic, group.max() + 1


# ---------------------------------------------------------------- 3. 线性探针

def variance_explained(X, design):
    """X 的每一维都用 design 做线性回归，算 X 的总变化里有多少能被 design 解释（0~1）。"""
    if len(X) != len(design) or not np.isfinite(X).all():
        raise ValueError("invalid variance-explanation inputs")
    Z = StandardScaler().fit_transform(X)
    D = np.hstack([design, np.ones((len(design), 1))])
    coef, *_ = np.linalg.lstsq(D, Z, rcond=None)
    total = (Z ** 2).sum()
    return float("nan") if total == 0 else 1 - ((Z - D @ coef) ** 2).sum() / total


def probe(X, counts, edge_ops, onehot, n_train=2000, n_test=5000):
    """线性探针：只用线性模型从特征里"读"信息，读得出来说明特征里清楚地写着这个信息。"""
    if len(X) <= n_train or n_train < 1 or n_test < 1:
        raise ValueError("probe needs disjoint nonempty train/test sets")
    perm = np.random.default_rng(0).permutation(len(X))
    tr, te = perm[:n_train], perm[n_train:n_train + n_test]
    reader = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    count_r2 = reader.fit(X[tr], counts[tr]).score(X[te], counts[te])
    edge_acc = []
    for e in range(6):  # 6 条边，每条边是 5 选 1 的分类题，瞎猜的准确率是 0.2
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        edge_acc.append(clf.fit(X[tr], edge_ops[tr, e]).score(X[te], edge_ops[te, e]))
    return dict(count_R2=count_r2, edge_acc_mean=np.mean(edge_acc), edge_acc_min=np.min(edge_acc),
                var_by_counts=variance_explained(X, counts), var_by_onehot=variance_explained(X, onehot))


# ---------------------------------------------------------------- 汇总和画图

def paired(df, target, head, a, b):
    from statistics_report import paired_text
    return paired_text(df, target, head, a, b)


def plot(df, ceilings):
    fig, axes = plt.subplots(len(TARGETS), len(HEADS), figsize=(16, 12), sharex=True)
    for r, target in enumerate(TARGETS):
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            sub = df[(df.target == target) & (df["head"] == head)]
            for feat, style in STYLE.items():
                g = sub[sub.features == feat]
                if g.empty:
                    continue
                stat = g.groupby("n_train")["tau"].agg(["mean", "std"])
                ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], marker="o", capsize=3, **style)
            ax.axhline(ceilings[target], color="tab:green", linestyle=":", linewidth=1.5,
                       label="op-count group-mean reference")
            ax.set_xscale("log")
            ax.set_title(f"{target}  |  head = {head}")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Kendall tau-b")
            if r == len(TARGETS) - 1:
                ax.set_xlabel("number of training architectures")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_counts_vs_llm.png", dpi=150)


def main():
    global FEATURE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=max(TRAIN_SIZES), help="只跑不超过这个数的训练样本数（冒烟测试用）")
    parser.add_argument("--out", default=str(HERE / "results_v3" / "step1"))
    parser.add_argument("--feature-dir", default=None)
    args = parser.parse_args()
    if args.feature_dir:
        FEATURE_DIR = Path(args.feature_dir)
    global OUT_DIR
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [n for n in TRAIN_SIZES if n <= args.max_n]

    archs, _ = load_dataset()
    feats = build_features(archs)
    oh, counts = feats["onehot"], feats["counts"]
    edge_ops = oh.reshape(-1, 6, 5).argmax(axis=2)

    # 2 和 3 很快，先算，几分钟内就能看到
    lines = ["===== 只靠操作计数的组均值参考（全部 15625 个结构，上帝视角）====="]
    ceilings = {}
    for target in TARGETS:
        y = load_dataset(target)[1]
        ceilings[target], n_groups = group_mean_reference(counts, y)
        lines.append(f"  {target:15s} τ 参考 = {ceilings[target]:.3f}（{n_groups} 种不同的计数组合）")

    lines += ["", "===== 线性探针（2000 个结构训练读取器，另外 5000 个测试）=====",
              "  count_R2      从特征读出 5 个操作计数的 R²（1 = 完全读得出）",
              "  edge_acc      读出每条边放了什么操作的准确率（6 条边的平均 / 最差，瞎猜 = 0.2）",
              "  var_by_*      样本内描述性拟合R²（不是留出泛化分数）", ""]
    probes = pd.DataFrame({name: probe(feats[name], counts, edge_ops, oh) for name in LLMS_ALL}).T
    lines.append(probes.round(3).to_string())
    text = "\n".join(lines)
    print(text + "\n", flush=True)

    df = learning_curves(feats, args.seeds, args.jobs, sizes)

    lines += ["", f"===== 学习曲线：Kendall τ-b 平均值（{args.seeds} 个种子）====="]
    order = HAND + LLMS_ALL
    for target in TARGETS:
        for head in HEADS:
            sub = df[(df.target == target) & (df["head"] == head)]
            table = sub.groupby(["features", "n_train"]).tau.mean().unstack()
            table = table.loc[[f for f in order if f in table.index]]
            lines += ["", f"[{target} | {head}]   计数组均值参考 = {ceilings[target]:.3f}", table.round(3).to_string()]

    lines += ["", "===== 配对比较：差值（前者赢的次数，Wilcoxon p）====="]
    for target in TARGETS:
        for head in HEADS:
            lines += ["", f"[{target} | {head}]",
                      "  计数 − 随机Pythia : " + paired(df, target, head, "counts", "pythia410m_step0"),
                      "  计数 − one-hot    : " + paired(df, target, head, "counts", "onehot"),
                      "  计数+onehot − one-hot : " + paired(df, target, head, "counts+onehot", "onehot")]

    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    plot(df, ceilings)
    print("\n" + text)
    print(f"\n结果已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
