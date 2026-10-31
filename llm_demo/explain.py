"""第 3 步（解释实验）：LLM 特征是不是主要记住了"每种操作各用了几次"？

背景：全空间实验发现，LLM 特征小样本时比 one-hot 好，样本多了却卡在 τ≈0.64 左右。
先导实验的猜想：LLM 特征主要编码了"操作计数"——5 种操作各出现在几条边上（5 个数），
而不太在意它们各在哪条边上。这个脚本从三个方面检验这个猜想:

1. 学习曲线：把"操作计数"（5 维）和"计数 + one-hot"（35 维）也当成特征来源，
   和 LLM 特征、one-hot 放在一起比，三个数据集 × 三种回归头 × 全部样本数 × 20 个种子
2. 计数的天花板：如果只知道操作计数，最好能排得多准？（同一种计数组合里的结构，
   只能给出同一个预测值，取它们真实精度的平均值，这是"上帝视角"的最好情况）
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
    oh = np.array([one_hot(a) for a in archs])
    feats = {"onehot": oh, "counts": op_counts(oh), "counts+onehot": np.hstack([op_counts(oh), oh])}
    for name in LLMS_ALL:
        feats[name] = np.load(FEATURE_DIR / f"{name}.npy")
    return feats


# ---------------------------------------------------------------- 1. 学习曲线

def run_one(X, y, n, seed, head):
    """一次实验：用种子 seed 抽 n 个结构训练，其余全部当测试集。和 evaluate.py 的抽法完全一样。"""
    perm = np.random.default_rng(seed).permutation(len(y))
    tr, te = perm[:n], perm[n:]
    model = make_head(head, seed)
    if head == "rf":  # 外面已经多进程并行了，每棵森林内部就别再开多线程，免得抢 CPU
        model.set_params(regressor__randomforestregressor__n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 不收敛提示、预测值全部相同时 tau 无定义之类
        pred = model.fit(X[tr], y[tr]).predict(X[te])
        tau = kendalltau(pred, y[te]).statistic
    return (0.0 if np.isnan(tau) else float(tau)), n_at_k(pred, y[te])


def learning_curves(feats, n_seeds, n_jobs, sizes):
    # 要算的组合：CIFAR-10 上所有特征；另两个数据集上手工特征 + 3 个代表性 LLM
    plan = [(t, f) for t in TARGETS for f in (HAND + LLMS_ALL if t == "cifar10_acc" else HAND + LLMS_MAIN)]

    # 已经算过的直接拿来用（evaluate.py 的结果，以及本脚本上次跑了一半的结果）
    partial_file = OUT_DIR / "curves_partial.csv"
    prev = pd.read_csv(partial_file) if partial_file.exists() else pd.DataFrame()
    old = pd.concat([pd.read_csv(HERE / "results" / "results.csv"), prev])
    old = old[old.seed < n_seeds]
    have = old.groupby(["target", "features", "head", "n_train"]).seed.nunique()

    jobs = [(t, f, h, n) for t, f in plan for h in HEADS for n in sizes
            if have.get((t, f, h, n), 0) < n_seeds]
    reused = old[[(r.target, r.features) in set(plan) and r.n_train in sizes for r in old.itertuples()]]
    total = len(plan) * len(HEADS) * len(sizes)
    print(f"学习曲线：一共 {total} 组，其中 {total - len(jobs)} 组之前算过直接复用，"
          f"这次要算 {len(jobs)} 组（每组 {n_seeds} 个种子，同时用 {n_jobs} 个进程）\n", flush=True)

    ys = {t: load_dataset(t)[1] for t in TARGETS}
    rows, t0 = [], time.time()
    # 用同一组进程跑完所有任务，省去反复启动进程的时间
    with Parallel(n_jobs=n_jobs) as pool:
        for k, (target, feat, head, n) in enumerate(jobs, 1):
            out = pool(delayed(run_one)(feats[feat], ys[target], n, s, head) for s in range(n_seeds))
            new = [dict(target=target, n_train=n, seed=s, features=feat, head=head, tau=tau, n_at_10=n10)
                   for s, (tau, n10) in enumerate(out)]
            rows += new
            # 随时存盘（连同上次没跑完的部分），中途停了下次能接着跑
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(partial_file, index=False)
            bar = "█" * int(20 * k / len(jobs))
            print(f"[{k:3d}/{len(jobs)}] {bar:<20s} {time.strftime('%H:%M')}  {target:15s} {feat:22s} "
                  f"{head:5s} n={n:<5d} τ={np.mean([r['tau'] for r in new]):.3f}   "
                  f"已用 {(time.time() - t0) / 60:.0f} 分钟", flush=True)

    df = pd.concat([reused, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates(["target", "features", "head", "n_train", "seed"])
    df.to_csv(OUT_DIR / "curves.csv", index=False)
    return df


# ---------------------------------------------------------------- 2. 计数的天花板

def count_ceiling(counts, y):
    """只知道操作计数时的最好排序：同一种计数组合的结构，都预测为这组结构的真实平均精度。"""
    _, group = np.unique(counts, axis=0, return_inverse=True)
    group = group.ravel()
    group_mean = np.bincount(group, weights=y) / np.bincount(group)
    return kendalltau(group_mean[group], y).statistic, group.max() + 1


# ---------------------------------------------------------------- 3. 线性探针

def variance_explained(X, design):
    """X 的每一维都用 design 做线性回归，算 X 的总变化里有多少能被 design 解释（0~1）。"""
    Z = StandardScaler().fit_transform(X)
    D = np.hstack([design, np.ones((len(design), 1))])
    coef, *_ = np.linalg.lstsq(D, Z, rcond=None)
    return 1 - ((Z - D @ coef) ** 2).sum() / (Z ** 2).sum()


def probe(X, counts, edge_ops, onehot, n_train=2000, n_test=5000):
    """线性探针：只用线性模型从特征里"读"信息，读得出来说明特征里清楚地写着这个信息。"""
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
    """同一批训练划分下，特征 a 减特征 b 的 tau 差：平均差、a 赢的次数、Wilcoxon p 值。"""
    out = []
    sub = df[(df.target == target) & (df["head"] == head)]
    for n in sorted(sub.n_train.unique()):
        x = sub[(sub.features == a) & (sub.n_train == n)].sort_values("seed").tau.values
        z = sub[(sub.features == b) & (sub.n_train == n)].sort_values("seed").tau.values
        if len(x) == 0 or len(x) != len(z):
            continue
        d = x - z
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out.append(f"n={n}: {d.mean():+.3f}（{(d > 0).sum()}/{len(d)}，p={p:.2g}）")
    return "；".join(out)


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
                       label="ceiling of op counts")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=max(TRAIN_SIZES), help="只跑不超过这个数的训练样本数（冒烟测试用）")
    args = parser.parse_args()
    sizes = [n for n in TRAIN_SIZES if n <= args.max_n]

    archs, _ = load_dataset()
    feats = build_features(archs)
    oh, counts = feats["onehot"], feats["counts"]
    edge_ops = oh.reshape(-1, 6, 5).argmax(axis=2)

    # 2 和 3 很快，先算，几分钟内就能看到
    lines = ["===== 只靠操作计数的天花板（全部 15625 个结构，上帝视角）====="]
    ceilings = {}
    for target in TARGETS:
        y = load_dataset(target)[1]
        ceilings[target], n_groups = count_ceiling(counts, y)
        lines.append(f"  {target:15s} τ 上限 = {ceilings[target]:.3f}（{n_groups} 种不同的计数组合）")

    lines += ["", "===== 线性探针（2000 个结构训练读取器，另外 5000 个测试）=====",
              "  count_R2      从特征读出 5 个操作计数的 R²（1 = 完全读得出）",
              "  edge_acc      读出每条边放了什么操作的准确率（6 条边的平均 / 最差，瞎猜 = 0.2）",
              "  var_by_*      特征的总变化里，能被计数 / 完整 one-hot 线性解释的比例", ""]
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
            lines += ["", f"[{target} | {head}]   计数天花板 = {ceilings[target]:.3f}", table.round(3).to_string()]

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
