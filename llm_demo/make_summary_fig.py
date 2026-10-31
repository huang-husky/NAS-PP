"""一张总结图：三个搜索空间放在一起，看"小样本 LLM 好、样本多有上限、词频能复现"是否都成立。

只画随机森林回归头（结论最清楚的那个），每个空间一张子图:
    one-hot（手工编码）、随机初始化 Pythia、训练完成的 Pythia、词频（bag-of-words）、只数操作的天花板
TransNAS 有 7 个任务，这里画 7 个任务的平均值。

运行方式（在 llm_demo 文件夹下，step2.py 跑完之后）:
    python make_summary_fig.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DIR = HERE / "results" / "step2"

PANELS = [  # (文件, 目标, 标题, 只数操作的天花板)
    ("curves_nb201.csv", ["cifar10_acc"], "NAS-Bench-201 (CIFAR-10, 15625 archs)", 0.629),
    ("curves_transnas.csv", None, "TransNAS-Bench-101 micro (mean of 7 tasks)", 0.509),
    ("curves_nb101.csv", ["test_acc"], "NAS-Bench-101 (5000 random archs)", 0.462),
]
STYLE = {
    "onehot": dict(color="black", linewidth=2.5, label="one-hot (hand-crafted)"),
    "pythia410m_step0": dict(color="tab:blue", linewidth=2, label="Pythia-410M, random init"),
    "pythia410m_step143000": dict(color="tab:red", linewidth=1.5, label="Pythia-410M, fully trained"),
    "bow": dict(color="tab:brown", linewidth=2.5, linestyle="-.", label="token counts of the code (bag of words)"),
}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (fname, targets, title, ceiling) in zip(axes, PANELS):
        df = pd.read_csv(DIR / fname)
        df = df[df["head"] == "rf"]
        if targets:
            df = df[df.target.isin(targets)]
        for feat, st in STYLE.items():
            g = df[df.features == feat]
            # 先在每个任务里按样本数取平均，再在任务之间取平均（TransNAS 用）
            curve = g.groupby(["target", "n_train"]).tau.mean().groupby("n_train").mean()
            ax.plot(curve.index, curve.values, marker="o", markersize=4, **st)
        ax.axhline(ceiling, color="tab:green", linestyle=":", linewidth=2, label="ceiling of op counts only")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("number of training architectures")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall tau-b (random forest head)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Frozen LLM features ≈ token counts of the code: strong with few samples, capped with many",
                 fontsize=13)
    fig.tight_layout()
    out = DIR / "fig_summary_3spaces.png"
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
