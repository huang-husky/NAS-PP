"""规律 2 的评估：每种写法下，LLM 特征的表现是不是 ≈ 这种写法的词频特征的表现？

特征来源（每个空间都一样）:
    onehot                              手工编码（参照）
    <写法>_bow                          这种写法的词频
    <写法>_pythia410m_step0             随机初始化 Pythia 的平均特征
    <写法>_pythia410m_step143000        训练完成 Pythia 的平均特征
写法见 formats.py：code / edges / sorted / nl / posword。

为了一晚上跑完：回归头只用岭回归和随机森林（MLP 没调参、结果不稳）；
TransNAS 只取 3 个任务（场景分类、语义分割、自编码器，其中自编码器是"计数最没用"的那个）；
训练集最多 1000 个；n ≤ 500 时 20 个种子，n = 1000 时 10 个。

主图 fig_formats_scatter.png：横轴是词频特征的 τ，纵轴是 LLM 特征的 τ（随机森林），
每个点是"写法 × 空间 × 样本数"。规律成立的话，点应该落在对角线附近。

运行方式（在 llm_demo 文件夹下）:
    python step3_formats.py --seeds 2 --max-n 50 --out results/_smoke3     冒烟测试
    python step3_formats.py --wait        等特征全部提取完再开始（每分钟检查一次）
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from arch_text import load_dataset
from embed import FEATURE_DIR, load_archs
from explain import count_ceiling, run_one
from formats import FORMATS
from nb101_text import encode_nb101, load_nb101
from progress import Progress
from step2 import NB_OPS, TN_OPS, TRANSNAS_CSV, encode, est_cost, paired_common, seeds_for

HERE = Path(__file__).parent
FMT_DIR = FEATURE_DIR / "formats"
HEADS = ["ridge", "rf"]
SIZES = [10, 50, 100, 500, 1000]
TARGETS = {"nb201": ["cifar10_acc"], "transnas": ["class_scene", "segmentsemantic", "autoencoder"],
           "nb101": ["test_acc"]}
# 规律 3：不同大小的 Pythia，每个都有随机初始化（step0）和训练完成（step143000）两个版本
SIZE_M = {"pythia70m": 70, "pythia160m": 160, "pythia410m": 410, "pythia1b": 1000}
ALL_LLMS = [f"{s}_{v}" for s in SIZE_M for v in ["step0", "step143000"]]
LLMS, FEATS = [], []


def set_models(names):
    """确定这次要评估哪些 LLM（main 里调用一次）。"""
    global LLMS, FEATS
    LLMS = names
    FEATS = ["onehot"] + [f"{fmt}_{k}" for fmt in FORMATS for k in ["bow", *LLMS]]


def label(llm):
    """pythia70m_step0 -> 'Pythia-70M, random'"""
    if llm == "qwen25_0.5b":
        return "Qwen2.5-0.5B, trained"
    size, step = llm.rsplit("_", 1)
    return f"Pythia-{size.removeprefix('pythia').upper()}, {'random' if step == 'step0' else 'trained'}"


set_models(["pythia410m_step0", "pythia410m_step143000"])
FMT_COLOR = {"code": "tab:blue", "edges": "tab:orange", "sorted": "tab:green", "nl": "tab:purple",
             "posword": "tab:red"}
SPACE_MARKER = {"nb201": "o", "transnas": "s", "nb101": "^"}


def load_space(space):
    """返回 (结构池里的结构, {目标: 标签}, one-hot, 操作计数)。"""
    pool = np.load(FMT_DIR / space / "pool.npy")
    archs = load_archs(space)
    archs = [archs[i] for i in pool]
    if space == "nb201":
        ys = {"cifar10_acc": load_dataset("cifar10_acc")[1][pool]}
        oh, cnt, _ = encode(archs, NB_OPS)
    elif space == "transnas":
        df = pd.read_csv(TRANSNAS_CSV)
        ys = {t: df[f"{t}_valid"].values.astype(float)[pool] for t in TARGETS[space]}
        oh, cnt, _ = encode(archs, TN_OPS)
    else:
        rows = load_nb101()[1]
        ys = {"test_acc": np.array([float(rows[i]["test_acc"]) for i in pool])}
        pairs = [encode_nb101(a) for a in archs]
        oh, cnt = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    return ys, oh, cnt


def needed_files():
    return [FMT_DIR / s / f"{f}.npy" for s in TARGETS for f in FEATS if f != "onehot"]


def curves(space, ys, oh, sizes, n_seeds, n_jobs, out_dir):
    partial = out_dir / f"curves_{space}_partial.csv"
    prev = pd.read_csv(partial) if partial.exists() else pd.DataFrame(
        columns=["target", "n_train", "seed", "features", "head", "tau", "n_at_10"])
    prev = prev[prev.seed < n_seeds]
    have = prev.groupby(["target", "features", "head", "n_train"]).seed.nunique()
    feats = [f for f in FEATS if f == "onehot" or (FMT_DIR / space / f"{f}.npy").exists()]  # 冒烟测试时有些还没提完
    jobs = [(f, t, h, n) for f in feats for t in ys for h in HEADS for n in sizes
            if have.get((t, f, h, n), 0) < seeds_for(n, n_seeds)]
    print(f"\n[{space}] 一共 {len(FEATS) * len(ys) * len(HEADS) * len(sizes)} 组，这次要算 {len(jobs)} 组", flush=True)

    def load(f):
        return oh if f == "onehot" else np.load(FMT_DIR / space / f"{f}.npy")

    dims = {f: (oh.shape[1] if f == "onehot" else np.load(FMT_DIR / space / f"{f}.npy", mmap_mode="r").shape[1])
            for f in dict.fromkeys(j[0] for j in jobs)}
    rows = []
    if jobs:
        bar = Progress(sum(est_cost(h, n, dims[f]) * seeds_for(n, n_seeds) for f, _, h, n in jobs),
                       name=space, every=120, units=False)
        with Parallel(n_jobs=n_jobs) as pool:
            for feat in dims:
                X = load(feat)
                for f, target, head, n in [j for j in jobs if j[0] == feat]:
                    k = seeds_for(n, n_seeds)
                    out = pool(delayed(run_one)(X, ys[target], n, s, head) for s in range(k))
                    rows += [dict(target=target, n_train=n, seed=s, features=feat, head=head, tau=t, n_at_10=n10)
                             for s, (t, n10) in enumerate(out)]
                    pd.concat([prev, pd.DataFrame(rows)]).to_csv(partial, index=False)
                    bar.update(est_cost(head, n, X.shape[1]) * k, note=f"{feat} {target} {head} n={n}")
    df = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    df = df.astype({"tau": float, "n_at_10": float, "n_train": int, "seed": int})
    return df.drop_duplicates(["target", "features", "head", "n_train", "seed"])


def agreement(all_df):
    """每个模型、每种回归头：LLM τ 和词频 τ 对得有多好（所有"写法 × 空间 × 任务 × 样本数"的点）。"""
    rows = []
    for llm in LLMS:
        for head in HEADS:
            xs, ys, fmts = [], [], []
            for df in all_df.values():
                mean = df[df["head"] == head].groupby(["target", "features", "n_train"]).tau.mean()
                for fmt in FORMATS:
                    for (target, n), x in mean.xs(f"{fmt}_bow", level="features").items():
                        y = mean.get((target, f"{fmt}_{llm}", n))
                        if y is not None:
                            xs.append(x), ys.append(y), fmts.append(fmt)
            if len(xs) < 3:
                continue
            xs, ys, fmts = np.array(xs), np.array(ys), np.array(fmts)
            keep = fmts != "posword"
            rows.append(dict(model=llm, head=head, points=len(xs), r=np.corrcoef(xs, ys)[0, 1],
                             mean_abs_gap=np.abs(ys - xs).mean(), mean_gap=(ys - xs).mean(),
                             r_no_posword=np.corrcoef(xs[keep], ys[keep])[0, 1]))
    return pd.DataFrame(rows)


def plot_sizes(agree, out_dir):
    """规律 3 的图：横轴模型大小，纵轴 LLM 和词频的吻合程度（相关系数、平均差距）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for head, ls in [("rf", "-"), ("ridge", "--")]:
        for step, color in [("step0", "tab:blue"), ("step143000", "tab:red")]:
            sub = agree[(agree["head"] == head) & agree.model.str.endswith(step)]
            if sub.empty:
                continue
            size = [SIZE_M[m.rsplit("_", 1)[0]] for m in sub.model]
            lab = f"{'random' if step == 'step0' else 'trained'}, {head}"
            axes[0].plot(size, sub.r, marker="o", color=color, linestyle=ls, label=lab)
            axes[1].plot(size, sub.mean_abs_gap, marker="o", color=color, linestyle=ls, label=lab)
    for ax, t in zip(axes, ["correlation between LLM tau and token-count tau", "mean |LLM tau - token-count tau|"]):
        ax.set_xscale("log")
        ax.set_xticks(list(SIZE_M.values()), [f"{v}M" if v < 1000 else "1B" for v in SIZE_M.values()])
        ax.set_xlabel("Pythia size (parameters)")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)  # 固定纵轴范围，免得很小的波动看起来像大起大落
    axes[1].set_ylim(0, 0.1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the token-count law hold for every model size?")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sizes.png", dpi=150)


def plot_scatter(all_df, out_dir):
    """主图：横轴词频 τ，纵轴 LLM τ，每个模型一张子图（随机森林）。上排随机初始化，下排训练完成。"""
    sizes = [s for s in dict.fromkeys(m.rsplit("_", 1)[0] for m in LLMS) if s in SIZE_M]
    others = [m for m in LLMS if m.rsplit("_", 1)[0] not in SIZE_M]  # 非 Pythia 的模型（如 Qwen）各占一列，放在上排
    cells = [(row, col, f"{size}_{step}") for col, size in enumerate(sizes)
             for row, step in enumerate(["step0", "step143000"])]
    cells += [(0, len(sizes) + i, m) for i, m in enumerate(others)]
    ncol = len(sizes) + len(others)
    fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol + 1, 10.5), sharex=True, sharey=True, squeeze=False)
    for i in range(len(others)):
        axes[1, len(sizes) + i].set_visible(False)
    for row, col, llm in cells:
        ax = axes[row, col]
        title = label(llm)
        if llm not in LLMS:
            ax.set_visible(False)
            continue
        for space, df in all_df.items():
            sub = df[df["head"] == "rf"]
            mean = sub.groupby(["target", "features", "n_train"]).tau.mean()
            for fmt in FORMATS:
                for (target, n), x in mean.xs(f"{fmt}_bow", level="features").items():
                    y = mean.get((target, f"{fmt}_{llm}", n))
                    if y is None:
                        continue
                    ax.scatter(x, y, color=FMT_COLOR[fmt], marker=SPACE_MARKER[space],
                               s=18 + 12 * np.log10(n) ** 2, alpha=0.8, edgecolor="white", linewidth=0.5)
        lim = [0, 0.85]
        ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_title(f"{title} (random forest head)", fontsize=10)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel("Kendall tau of token-count features")
        if col == 0:
            ax.set_ylabel("Kendall tau of LLM features")
    axes = axes.ravel()
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=f) for f, c in FMT_COLOR.items()]
    handles += [plt.Line2D([], [], color="gray", marker=m, linestyle="", label=s) for s, m in SPACE_MARKER.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", title="format / space (bigger = more samples)",
                   title_fontsize=8)
    fig.suptitle("Does the text format's token counts predict what the LLM features can do?")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formats_scatter.png", dpi=150)


def plot_curves(all_df, out_dir):
    """每个空间一列：随机 Pythia 在五种写法下的学习曲线（实线）、对应的词频（虚线）和 one-hot（黑线）。"""
    cols = [(s, t) for s, df in all_df.items() for t in TARGETS[s]]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.6), sharex=True)
    for ax, (space, target) in zip(np.atleast_1d(axes), cols):
        sub = all_df[space]
        sub = sub[(sub["head"] == "rf") & (sub.target == target)]
        g = sub.groupby(["features", "n_train"]).tau.mean()
        ax.plot(g["onehot"].index, g["onehot"].values, color="black", linewidth=2.5, label="one-hot")
        for fmt in FORMATS:
            for feat, ls in [(f"{fmt}_pythia410m_step0", "-"), (f"{fmt}_bow", ":")]:
                if feat in g.index.get_level_values(0):
                    ax.plot(g[feat].index, g[feat].values, color=FMT_COLOR[fmt], linestyle=ls, marker="o",
                            markersize=3, label=f"{fmt} ({'random Pythia' if ls == '-' else 'token counts'})")
        ax.set_xscale("log")
        ax.set_title(f"{space} | {target}", fontsize=10)
        ax.set_xlabel("n train")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Kendall tau (random forest)")
    np.atleast_1d(axes)[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formats_curves.png", dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=max(SIZES))
    parser.add_argument("--out", default=str(HERE / "results" / "step3"))
    parser.add_argument("--wait", action="store_true", help="等所有特征文件都出现后再开始")
    parser.add_argument("--models", default="pythia410m_step0,pythia410m_step143000",
                        help="要评估的 LLM，逗号隔开；all = 4 种大小 × 2 个版本")
    args = parser.parse_args()
    set_models(ALL_LLMS if args.models == "all" else args.models.split(","))
    sizes = [n for n in SIZES if n <= args.max_n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wait:
        while missing := [f for f in needed_files() if not f.exists()]:
            print(f"{time.strftime('%H:%M')} 还差 {len(missing)} 个特征文件，1 分钟后再看", flush=True)
            time.sleep(60)
        time.sleep(30)  # 最后一个文件刚写出来，留点时间写完

    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    all_df = {}
    for space in TARGETS:
        ys, oh, cnt = load_space(space)
        say("", f"===== [{space}] 天花板（{len(oh)} 个结构；同一组只能给同一个预测，用真实精度的组平均，上帝视角）=====")
        for target, y in ys.items():
            c, g = count_ceiling(cnt, y)
            say(f"  {target:16s} 操作计数：τ 上限 {c:.3f}（{g} 组）")
            for fmt in FORMATS:
                c, g = count_ceiling(np.load(FMT_DIR / space / f"{fmt}_bow.npy"), y)
                say(f"  {target:16s} {fmt:8s} 词频：τ 上限 {c:.3f}（{g} 组）")

        df = curves(space, ys, oh, sizes, args.seeds, args.jobs, out_dir)
        df.to_csv(out_dir / f"curves_{space}.csv", index=False)
        all_df[space] = df

        say("", f"===== [{space}] Kendall τ-b 平均值 =====")
        for target in ys:
            for head in HEADS:
                sub = df[(df.target == target) & (df["head"] == head)]
                table = sub.groupby(["features", "n_train"]).tau.mean().unstack()
                say("", f"[{target} | {head}]", table.loc[[f for f in FEATS if f in table.index]].round(3).to_string())

        say("", f"===== [{space}] 配对比较：差值（前者赢的次数，Wilcoxon p）=====")
        for target in ys:
            for head in HEADS:
                out = [f"[{target} | {head}]"]
                for fmt in FORMATS:
                    for llm in LLMS:
                        out.append(f"  {fmt}_{llm} − {fmt}_bow : " +
                                   paired_common(df, target, head, f"{fmt}_{llm}", f"{fmt}_bow"))
                for llm in LLMS:
                    out.append(f"  posword_{llm} − code_{llm} : " +
                               paired_common(df, target, head, f"posword_{llm}", f"code_{llm}"))
                    out.append(f"  posword_{llm} − onehot : " + paired_common(df, target, head, f"posword_{llm}", "onehot"))
                say("", *out)

    agree = agreement(all_df)
    say("", "===== 规律 3：每个模型的 LLM τ 和词频 τ 吻合得怎么样（所有写法 × 空间 × 任务 × 样本数）=====",
        "  r = 相关系数；mean_abs_gap = 平均差距；mean_gap = 平均偏差（LLM − 词频）；r_no_posword = 去掉 posword 后的相关系数",
        agree.round(3).to_string(index=False))
    plot_sizes(agree, out_dir)
    plot_scatter(all_df, out_dir)
    plot_curves(all_df, out_dir)
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
