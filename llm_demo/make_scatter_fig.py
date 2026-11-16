"""画备用页的散点图：每个“写法 × 任务”组合下，LLM 特征的 τ 对比同一写法词频的 τ（Pythia-410M，n=1000）。

数据来源：results_repaired/step3/curves_*_quality.csv（格式实验，修复后的结果），回归头为随机森林。
每张图 25 个点（5 种写法 × NB201 CIFAR-10、TransNAS 3 个任务、NB101）。不跑新实验。
画图用的数字同时存成 CSV，方便复查。

运行方式（在 llm_demo 文件夹下）:
    python make_scatter_fig.py
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "results_repaired" / "step3"
OUT = HERE / "results_v3" / "model_sizes"
N = 1000
FMT_COLOR = {"code": "#1f77b4", "edges": "#ff7f0e", "sorted": "#2ca02c", "nl": "#9467bd", "posword": "#d62728"}
SPACE_MARK = {"nb201": "o", "transnas": "s", "nb101": "^"}
MODELS = [("pythia410m_step0", "Random Pythia-410M"), ("pythia410m_step143000", "Trained Pythia-410M")]


def points():
    q = pd.concat([pd.read_csv(f).assign(space=Path(f).stem.split("_")[1])
                   for f in glob.glob(str(SRC / "curves_*_quality.csv"))])
    q = q[(q["head"] == "rf") & (q.n_train == N)]
    q["fmt"] = q.features.str.split("_").str[0]
    q["model"] = q.features.str.split("_", n=1).str[1]
    bow = q[q.model == "bow"][["space", "target", "fmt", "mean"]].rename(columns={"mean": "token_counts_tau"})
    m = q[q.model.isin([k for k, _ in MODELS])][["space", "target", "fmt", "model", "mean"]]
    m = m.rename(columns={"mean": "llm_tau"}).merge(bow, on=["space", "target", "fmt"])
    for k, _ in MODELS:
        assert (m.model == k).sum() == 25, (k, (m.model == k).sum())
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = points()
    m.sort_values(["model", "space", "target", "fmt"]).to_csv(OUT / "scatter_points_n1000.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=200, sharey=True)
    for ax, (key, title) in zip(axes, MODELS):
        g = m[m.model == key]
        for _, r in g.iterrows():
            ax.scatter(r.token_counts_tau, r.llm_tau, c=FMT_COLOR[r.fmt], marker=SPACE_MARK[r.space], s=45, alpha=0.85)
        ax.plot([0, 0.9], [0, 0.9], "--", color="gray", lw=1.2)
        ax.set_xlim(-0.05, 0.95)
        ax.set_ylim(-0.05, 0.95)
        ax.set_title(f"{title} (n={N})", fontsize=11)
        ax.set_xlabel("Token-count Kendall τ")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LLM-feature Kendall τ")
    shapes = [Line2D([], [], ls="", marker=mk, color="k", label=sp) for sp, mk in SPACE_MARK.items()]
    colors = [Line2D([], [], ls="", marker="o", color=c, label=f) for f, c in FMT_COLOR.items()]
    axes[1].legend(handles=shapes, title="Shape = search space", loc="upper left", fontsize=8, title_fontsize=8)
    fig.legend(handles=colors, title="Color = text format", loc="lower center", ncol=5, fontsize=9,
               title_fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_scatter_n1000.png", bbox_inches="tight")
    below = m[m.llm_tau < m.token_counts_tau - 0.05]
    print(f"已保存到 {OUT}。比词频低 0.05 以上的点：")
    print(below[["model", "space", "target", "fmt", "llm_tau", "token_counts_tau"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
