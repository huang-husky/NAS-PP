"""一张总结图：三个搜索空间放在一起，看"小样本 LLM 好、大样本表现、词频能复现"是否都成立。

只画随机森林回归头（结论最清楚的那个），每个空间一张子图:
    one-hot（手工编码）、随机初始化 Pythia、训练完成的 Pythia、词频（bag-of-words）、只数操作的组均值参考
TransNAS 有 7 个任务，这里画 7 个任务的平均值。

运行方式（在 llm_demo 文件夹下，step2.py 跑完之后）:
    python make_summary_fig.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from step2 import load_space, encode_any
from explain import group_mean_reference

HERE = Path(__file__).parent
DIR = HERE / "results_repaired" / "step2"

PANELS = [  # (文件, 目标, 标题); references are calculated from each current pool
    ("curves_nb201.csv", ["cifar10_acc"], "NAS-Bench-201 (CIFAR-10, 15625 archs)"),
    ("curves_transnas.csv", None, "TransNAS-Bench-101 micro (mean of 7 tasks)"),
    ("curves_nb101.csv", ["test_acc"], "NAS-Bench-101 (5000 random archs)"),
]
STYLE = {
    "onehot": dict(color="black", linewidth=2.5, label="one-hot (hand-crafted)"),
    "pythia410m_step0": dict(color="tab:blue", linewidth=2, label="Pythia-410M, random init"),
    "pythia410m_step143000": dict(color="tab:red", linewidth=1.5, label="Pythia-410M, fully trained"),
    "bow": dict(color="tab:brown", linewidth=2.5, linestyle="-.", label="token counts of the code (bag of words)"),
}


def main(counts_baseline=False):
    DIR.mkdir(parents=True, exist_ok=True)
    styles = dict(STYLE)
    if counts_baseline:
        styles["counts"] = dict(color="tab:green", linestyle="--", linewidth=2, label="op counts (trained RF baseline)")
    aggregates = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (fname, targets, title) in zip(axes, PANELS):
        space = fname.removeprefix("curves_").removesuffix(".csv")
        archs, ys, ops = load_space(space)
        _, counts = encode_any(space, archs, ops)
        values = [group_mean_reference(counts, y)[0] for t, y in ys.items()
                  if targets is None or t in targets]
        ceiling = sum(values) / len(values)
        df = pd.read_csv(DIR / fname)
        df = df[df["head"] == "rf"]
        if targets:
            df = df[df.target.isin(targets)]
        selected = df[df.features.isin(styles)]
        required_targets = targets or list(ys)
        if selected.duplicated(["target", "features", "n_train", "seed"]).any():
            raise ValueError("Duplicate curve records")
        complete = []
        for n, chunk in selected.groupby("n_train"):
            common = None
            for feature in styles:
                for target in required_targets:
                    seeds = set(chunk.loc[(chunk.features == feature) & (chunk.target == target) & chunk.tau.notna(), "seed"])
                    common = seeds if common is None else common & seeds
            if not common:
                raise ValueError(f"No complete common-seed comparison for {space}, n={n}")
            complete.append(chunk[chunk.seed.isin(common)])
        selected = pd.concat(complete, ignore_index=True)
        for feat, st in styles.items():
            g = selected[selected.features == feat]
            per_seed = g.groupby(["seed", "n_train"]).tau.mean()
            stat = per_seed.groupby("n_train").agg(["mean", "std", "count"])
            aggregates.append(stat.reset_index().assign(space=space, features=feat))
            ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], capsize=3, marker="o", markersize=4, **st)
        if not counts_baseline:
            ax.axhline(ceiling, color="tab:green", linestyle=":", linewidth=2,
                       label="op-count group-mean oracle (NOT an LLM limit)")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("number of training architectures")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall tau-b (random forest head)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Frozen LLM and token-count features: empirical learning curves (bars = seed SD)",
                 fontsize=13)
    fig.tight_layout()
    stem = "fig_summary_counts_baseline" if counts_baseline else "fig_summary_3spaces"
    out = DIR / (stem + ".png")
    pd.concat(aggregates, ignore_index=True).to_csv(DIR / (stem + "_points.csv"), index=False)
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reference mode is a descriptive oracle, NOT an LLM asymptote.")
    parser.add_argument("--counts-baseline", action="store_true", help="Plot actual trained count features instead of the oracle reference")
    args = parser.parse_args()
    main(args.counts_baseline)
