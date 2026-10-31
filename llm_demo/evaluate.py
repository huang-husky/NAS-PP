"""第 2 步：用提取好的特征训练预测器，比较排序能力，画图。

预测器 = 特征来源 + 回归头，两两组合:
    特征来源   onehot（手工编码，6 条边 x 5 种操作 = 30 维 0/1 向量）
               以及 features/ 里每个 LLM 的冻结特征
    回归头     ridge   岭回归（线性）
               rf      随机森林（非线性）
               mlp     两层全连接神经网络（非线性）
每个特征都配齐三种回归头，这样能分清"特征好不好"和"回归头强不强"。

训练集是从结构池里随机抽的 n 个结构，其余结构当测试集。

预测目标:
    cifar10_acc     真实精度
    params_M        参数量（合成目标：可以直接从结构算出来，不可能是"背"出来的）

指标:
    Kendall tau-b   整体排序对不对，越接近 1 越好（tau-b 能正确处理精度相同的并列情况）
    N@10            预测最好的 10 个结构里，真实排名最好的那个排第几，越小越好

运行方式（在 llm_demo 文件夹下）:
    python evaluate.py              用 features/ 里已有的全部特征
    python evaluate.py --seeds 3    先用 3 个种子快速看一眼
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arch_text import load_dataset, one_hot
from models import MODELS

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "features"
RESULT_DIR = HERE / "results"
RESULT_DIR.mkdir(exist_ok=True)

TARGETS = ["cifar10_acc", "params_M"]
TRAIN_SIZES = [10, 50, 100, 500, 1000, 2000]
HEADS = ["ridge", "rf", "mlp"]


def make_head(head, seed):
    """按名字造一个回归头。所有回归头都先把特征标准化，把预测目标也标准化（精度在 90 左右，
    神经网络直接学这么大的数很吃力），预测完再换算回原来的单位。"""
    if head == "ridge":
        # 线性回归 + 防过拟合的惩罚，惩罚强度用交叉验证自动选
        model = RidgeCV(alphas=np.logspace(-2, 5, 15))
    elif head == "rf":
        # 100 棵决策树投票；n_jobs=-1 表示用上所有 CPU 核
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    elif head == "mlp":
        # 两层各 64 个神经元；alpha 是防过拟合的惩罚。样本最少只有 10 个，所以不做早停
        model = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=1e-3,
                             max_iter=1000, random_state=seed)
    else:
        raise ValueError(head)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model),
                                      transformer=StandardScaler())


def n_at_k(pred, true, k=10):
    """预测排前 k 的结构里，真实名次最好的那个是第几名（1 = 测试集里真正的第一）。"""
    true_rank = np.argsort(np.argsort(-true)) + 1
    top_k = np.argsort(-pred)[:k]
    return int(true_rank[top_k].min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    pool = np.load(FEATURE_DIR / "pool.npy")
    archs, _ = load_dataset()

    # 特征来源：one-hot + 已经算好的 LLM 特征（按 models.py 里的顺序，没算好的就跳过）
    features = {"onehot": np.array([one_hot(archs[i]) for i in pool])}
    for name in MODELS:
        for suffix in ["", "_anon"]:
            f = FEATURE_DIR / f"{name}{suffix}.npy"
            if f.exists():
                features[name + suffix] = np.load(f)
    print(f"结构池 {len(pool)} 个；特征来源：{list(features)}；回归头：{HEADS}")

    # 训练集不能把结构池用光，至少给测试集留 200 个结构
    sizes = [n for n in TRAIN_SIZES if n <= len(pool) - 200]

    rows = []
    for target in TARGETS:
        _, y_all = load_dataset(target)
        y = y_all[pool]
        for n in sizes:
            for seed in range(args.seeds):
                perm = np.random.default_rng(seed).permutation(len(pool))
                tr, te = perm[:n], perm[n:]
                for feat_name, X in features.items():
                    for head in HEADS:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            pred = make_head(head, seed).fit(X[tr], y[tr]).predict(X[te])
                            warnings.simplefilter("ignore")  # 预测值全部相同时 tau 无定义，记为 0
                            tau = kendalltau(pred, y[te]).statistic
                        rows.append(dict(target=target, n_train=n, seed=seed, features=feat_name, head=head,
                                         tau=0.0 if np.isnan(tau) else tau, n_at_10=n_at_k(pred, y[te])))
            print(f"[完成] 目标 {target}，n = {n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "results.csv", index=False)

    # 汇总表：每种组合在每种样本数下 tau 的 平均值 ± 标准差
    keys = ["target", "head", "features", "n_train"]
    summary = df.groupby(keys)["tau"].agg(lambda s: f"{s.mean():.3f}±{s.std():.3f}").unstack("n_train")
    summary_n10 = df.groupby(keys)["n_at_10"].median().unstack("n_train")
    text = (f"结构池 {len(pool)} 个，每种设置 {args.seeds} 个种子\n\n"
            "Kendall tau-b（平均值±标准差）\n" + summary.to_string() + "\n\n"
            "N@10（中位数，越小越好）\n" + summary_n10.to_string() + "\n")
    (RESULT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    plot_curves(df)
    plot_dose(df)
    print(f"结果已保存到 {RESULT_DIR}")


def plot_curves(df):
    """图 1：tau 随训练样本数的变化。行 = 预测目标，列 = 回归头，每条线是一种特征来源。"""
    fig, axes = plt.subplots(len(TARGETS), len(HEADS), figsize=(15, 8), sharex=True)
    for r, target in enumerate(TARGETS):
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            sub = df[(df.target == target) & (df["head"] == head)]
            for feat, g in sub.groupby("features", sort=False):
                stat = g.groupby("n_train")["tau"].agg(["mean", "std"])
                style = dict(color="black", linewidth=2.5) if feat == "onehot" else {}
                ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], marker="o", capsize=3,
                            label=feat, **style)
            ax.set_xscale("log")
            ax.set_title(f"{target}  |  head = {head}")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Kendall tau-b")
            if r == len(TARGETS) - 1:
                ax.set_xlabel("number of training architectures")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig1_tau_vs_n.png", dpi=150)


def plot_dose(df):
    """图 2：Pythia 剂量曲线（预训练步数 -> tau），每个回归头一张子图，每种样本数一条线。"""
    steps = {f"pythia410m_step{s}": s for s in [0, 1000, 10000, 143000]}
    sub = df[df.features.isin(steps) & (df.target == "cifar10_acc")]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(HEADS), figsize=(15, 4.5), sharey=True)
    for ax, head in zip(axes, HEADS):
        for n, g in sub[sub["head"] == head].groupby("n_train"):
            stat = g.groupby("features")["tau"].mean()
            xs = np.array([steps[m] for m in stat.index])
            order = np.argsort(xs)
            ax.plot(xs[order] + 1, stat.values[order], marker="o", label=f"n = {n}")
        ax.set_xscale("log")
        ax.set_xlabel("Pythia-410M pretraining step (+1, log scale)")
        ax.set_title(f"CIFAR-10  |  head = {head}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall tau-b")
    axes[0].legend()
    fig.suptitle("Does more pretraining help?")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig2_pythia_dose.png", dpi=150)


if __name__ == "__main__":
    main()
