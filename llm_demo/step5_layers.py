"""规律 4 的评估：不同层、不同取法（mean / last）的特征，和词频比怎么样？

特征来源（每个空间）:
    onehot、code_bow（code 写法的词频）
    pythia410m_step0 / step143000 × 第 0、6、12、18、24 层 × mean / last
回归头：岭回归、随机森林。为了一晚上跑完，用 10 个种子（n = 1000 时 5 个）。
另外在 NB201 上做线性探针：每种特征里能读出多少"计数"和"每条边放了什么"。

主图 fig_layers.png：横轴是层，纵轴是 τ（随机森林），上排 n=10、下排 n=1000；
虚线是词频，黑线是 one-hot。

运行方式（在 llm_demo 文件夹下）:
    python step5_layers.py --seeds 2 --max-n 50 --out results/_smoke5 --no-probe     冒烟测试
    python step5_layers.py --wait
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step3_formats as S
from embed import FEATURE_DIR
from explain import probe
from run_layers import LAYERS, MODELS

HERE = Path(__file__).parent
LAYER_DIR = FEATURE_DIR / "layers"
POOLS = ["mean", "last"]
LAYER_FEATS = [f"{m}_L{l}_{p}" for m in MODELS for p in POOLS for l in LAYERS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=1000)
    parser.add_argument("--out", default=str(HERE / "results" / "step5"))
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--feat-dir", default=None, help="特征文件夹（冒烟测试用）")
    args = parser.parse_args()
    global LAYER_DIR
    if args.feat_dir:
        LAYER_DIR = Path(args.feat_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [n for n in S.SIZES if n <= args.max_n]

    # 借用规律 2 的评估代码，只是换成这里的特征文件夹和特征清单
    S.FMT_DIR = LAYER_DIR
    S.FEATS = ["onehot", "code_bow"] + LAYER_FEATS

    if args.wait:
        need = [LAYER_DIR / s / f"{f}.npy" for s in S.TARGETS for f in LAYER_FEATS]
        while missing := [f for f in need if not f.exists()]:
            print(f"{time.strftime('%H:%M')} 还差 {len(missing)} 个特征文件，1 分钟后再看", flush=True)
            time.sleep(60)
        time.sleep(30)

    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    all_df = {}
    for space in S.TARGETS:
        ys, oh, cnt = S.load_space(space)
        if space == "nb201" and not args.no_probe:
            say("", "===== [nb201] 线性探针（每种特征里能读出多少计数 / 每条边的操作；edge_acc 瞎猜 = 0.2）=====")
            edge_ops = oh.reshape(-1, 6, 5).argmax(axis=2)
            rows = {}
            for f in ["code_bow"] + LAYER_FEATS:
                X = np.load(LAYER_DIR / space / f"{f}.npy")
                if X.std(axis=0).max() == 0:  # 第 0 层的最后一个 token 对所有结构都一样，没法探针
                    continue
                rows[f] = probe(X, cnt, edge_ops, oh)
                print(f"    探针 {f} 完成", flush=True)
            say(pd.DataFrame(rows).T.round(3).to_string())

        df = S.curves(space, ys, oh, sizes, args.seeds, args.jobs, out_dir)
        df.to_csv(out_dir / f"curves_{space}.csv", index=False)
        all_df[space] = df
        say("", f"===== [{space}] Kendall τ-b 平均值 =====")
        for target in ys:
            for head in S.HEADS:
                sub = df[(df.target == target) & (df["head"] == head)]
                table = sub.groupby(["features", "n_train"]).tau.mean().unstack()
                say("", f"[{target} | {head}]", table.loc[[f for f in S.FEATS if f in table.index]].round(3).to_string())
        say("", f"===== [{space}] 配对比较：差值（前者赢的次数，Wilcoxon p）=====")
        for target in ys:
            for head in S.HEADS:
                out = [f"[{target} | {head}]"]
                for m in MODELS:
                    for l in LAYERS[1:]:
                        out.append(f"  {m}_L{l}_last − code_bow : " +
                                   S.paired_common(df, target, head, f"{m}_L{l}_last", "code_bow"))
                        out.append(f"  {m}_L{l}_mean − code_bow : " +
                                   S.paired_common(df, target, head, f"{m}_L{l}_mean", "code_bow"))
                say("", *out)

    plot(all_df, out_dir)
    print(f"\n结果已保存到 {out_dir}")


def plot(all_df, out_dir):
    cols = [(s, t) for s in all_df for t in S.TARGETS[s]]
    ns = [10, 1000]
    fig, axes = plt.subplots(len(ns), len(cols), figsize=(4.2 * len(cols), 8), squeeze=False)
    style = {("pythia410m_step0", "mean"): dict(color="tab:blue", marker="o", label="random, mean"),
             ("pythia410m_step0", "last"): dict(color="tab:blue", marker="s", linestyle="--", label="random, last token"),
             ("pythia410m_step143000", "mean"): dict(color="tab:red", marker="o", label="trained, mean"),
             ("pythia410m_step143000", "last"): dict(color="tab:red", marker="s", linestyle="--",
                                                     label="trained, last token")}
    for c, (space, target) in enumerate(cols):
        df = all_df[space]
        mean = df[(df["head"] == "rf") & (df.target == target)].groupby(["features", "n_train"]).tau.mean()
        for r, n in enumerate(ns):
            ax = axes[r, c]
            if (("onehot", n) not in mean.index):
                ax.set_visible(False)
                continue
            for (m, p), st in style.items():
                ys = [mean.get((f"{m}_L{l}_{p}", n), np.nan) for l in LAYERS]
                ax.plot(LAYERS, ys, **st)
            ax.axhline(mean[("code_bow", n)], color="tab:brown", linestyle=":", linewidth=2, label="token counts")
            ax.axhline(mean[("onehot", n)], color="black", linewidth=1.5, label="one-hot")
            ax.set_title(f"{space} | {target} | n={n}", fontsize=9)
            ax.set_xticks(LAYERS)
            ax.grid(alpha=0.3)
            if r == len(ns) - 1:
                ax.set_xlabel("layer (0 = token embeddings)")
            if c == 0:
                ax.set_ylabel("Kendall tau (random forest)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Pythia-410M, code format: which layer / pooling, and does it beat token counts?")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_layers.png", dpi=130)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
