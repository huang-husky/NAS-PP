"""画“模型大小”图：Pythia 70M / 160M / 410M / 1B 的 LLM 特征和词频有多一致（第 7 页左图）。

数据来源：results_repaired/step3/curves_*_quality.csv（格式实验，修复后的结果）。
计算方法和第 6 页一样：
    回归头固定为随机森林；在每个训练样本数 n 下，取 25 个“写法 × 任务”组合
    （5 种写法 × NB201 CIFAR-10、TransNAS 3 个任务、NB101），
    算 LLM 特征的平均 τ 和同一写法词频的平均 τ 之间的相关系数 r。
不跑任何新实验，只从已有结果计算。画图用的数字同时存成 CSV，方便复查。

运行方式（在 llm_demo 文件夹下）:
    python make_size_fig.py
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "results_repaired" / "step3"
OUT = HERE / "results_v3" / "model_sizes"
SIZES = [("70m", "70M"), ("160m", "160M"), ("410m", "410M"), ("1b", "1B")]
NS = [10, 100, 1000]


def agreement():
    """返回每个（模型大小, 是否训练过, n）的相关系数 r 和组合个数。"""
    q = pd.concat([pd.read_csv(f).assign(space=Path(f).stem.split("_")[1])
                   for f in glob.glob(str(SRC / "curves_*_quality.csv"))])
    q = q[q["head"] == "rf"]
    q["fmt"] = q.features.str.split("_").str[0]
    q["model"] = q.features.str.split("_", n=1).str[1]
    bow = q[q.model == "bow"][["space", "target", "n_train", "fmt", "mean"]].rename(columns={"mean": "bow"})
    m = q[q.model.str.startswith("pythia", na=False)].merge(bow, on=["space", "target", "n_train", "fmt"])
    rows = []
    for (model, n), g in m.groupby(["model", "n_train"]):
        size = model.replace("pythia", "").split("_")[0]
        rows.append(dict(size=size, trained=not model.endswith("step0"), n_train=int(n),
                         r=float(np.corrcoef(g["mean"], g["bow"])[0, 1]), points=len(g)))
    df = pd.DataFrame(rows)
    assert (df.points == 25).all(), "每个点应该是 25 个组合"
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = agreement()
    df.sort_values(["trained", "n_train", "size"]).to_csv(OUT / "agreement_by_size.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.9, 3.3), dpi=200)
    x = np.arange(len(SIZES))
    styles = {10: ("-", "o"), 100: ("--", "s"), 1000: (":", "^")}
    for trained, color, name in [(False, "#1f77b4", "Random"), (True, "#d62728", "Trained")]:
        for n in NS:
            g = df[(df.trained == trained) & (df.n_train == n)].set_index("size")
            y = [g.loc[s, "r"] for s, _ in SIZES]
            ls, mk = styles[n]
            ax.plot(x, y, ls, marker=mk, color=color, lw=1.6, ms=5, label=f"{name}, n={n}")
    ax.set_xticks(x, [lab for _, lab in SIZES])
    ax.set_xlabel("Pythia size (parameters)")
    ax.set_ylabel("r with token counts")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title("Agreement with token counts by model size (RF, 25 conditions)", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_sizes.png")
    print(df.pivot_table(index=["trained", "size"], columns="n_train", values="r").round(3).to_string())
    print(f"图和数字已保存到 {OUT}")


if __name__ == "__main__":
    main()
