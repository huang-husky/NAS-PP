"""规律 5：这些规律对"找最好的结构"有什么影响？

A. N@10 分析（用规律 2、3 已经算好的结果，不用重新训练）
   N@10 = 预测最好的 10 个结构里，真实名次最好的那个排第几（1 = 测试集里真正的第一，越小越好）。
   1) N@10 是不是也"跟着词频走"：每种写法下 LLM 特征的 N@10 和词频的 N@10 比
   2) "只看计数 / 词频"时 N@10 最好能到多少（上帝视角）：同一组的结构分不开，
      只能按组的真实平均精度排，组里随机挑；如果最好的结构所在的组很大，挑中它的机会就很小

B. 模拟搜索（更接近 NAS 实际怎么用预测器）
   先随机测 10 个结构 -> 用测过的结构训练预测器（随机森林）-> 预测结构池里所有没测过的结构 ->
   测预测最好的 10 个 -> 重复，直到一共测了 200 个。
   记录"到目前为止找到的最好结构，在整个结构池里排第几"（1 = 找到了池子里真正的最好结构）。
   比较的特征：one-hot、计数 + one-hot、code 写法的词频、code 写法的随机 / 训练完成 Pythia、
   posword 写法的随机 Pythia，以及完全随机搜索。每种 20 次（不同的初始 10 个结构）。

运行方式（在 llm_demo 文件夹下）:
    python step6_search.py --runs 2 --budget 40 --out results/_smoke6     冒烟测试
    python step6_search.py
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

import step3_formats as S
from evaluate import make_head
from progress import Progress

HERE = Path(__file__).parent
SEARCH_FEATS = ["onehot", "counts+onehot", "code_bow", "code_pythia410m_step0", "code_pythia410m_step143000",
                "posword_pythia410m_step0", "code_qwen25_0.5b", "random"]
LABEL = {"onehot": "one-hot", "counts+onehot": "counts + one-hot", "code_bow": "token counts (code)",
         "code_pythia410m_step0": "random Pythia (code)", "code_pythia410m_step143000": "trained Pythia (code)",
         "posword_pythia410m_step0": "random Pythia (posword)", "code_qwen25_0.5b": "Qwen2.5-0.5B (code)",
         "random": "random search"}
COLOR = {"onehot": "black", "counts+onehot": "tab:purple", "code_bow": "tab:brown", "code_pythia410m_step0": "tab:blue",
         "code_pythia410m_step143000": "tab:red", "posword_pythia410m_step0": "tab:orange",
         "code_qwen25_0.5b": "tab:green", "random": "gray"}


# ---------------------------------------------------------------- A. N@10

def oracle_n_at_10(groups, y, reps=300, seed=0):
    """只知道分组时 N@10 的最好情况：按组的真实平均精度从高到低挑满 10 个，最后一组里随机挑。"""
    rng = np.random.default_rng(seed)
    true_rank = np.argsort(np.argsort(-y)) + 1
    _, g = np.unique(groups, axis=0, return_inverse=True)
    g = g.ravel()
    order = np.argsort(-(np.bincount(g, weights=y) / np.bincount(g)))
    out = []
    for _ in range(reps):
        picked = []
        for k in order:
            members = np.flatnonzero(g == k)
            need = 10 - len(picked)
            picked += list(rng.choice(members, size=min(need, len(members)), replace=False))
            if len(picked) >= 10:
                break
        out.append(true_rank[picked].min())
    return float(np.median(out)), float(np.mean(np.array(out) == 1))


def n10_analysis(say):
    say("", "===== A. N@10（随机森林，中位数；越小越好）=====")
    rows = []
    for space in S.TARGETS:
        df = pd.read_csv(HERE / "results" / "step3" / f"curves_{space}.csv")
        df = df[df["head"] == "rf"]
        rows.append(df.assign(space=space))
    df = pd.concat(rows)
    med = df.groupby(["space", "target", "features", "n_train"]).n_at_10.median()
    show = ["onehot", "code_bow", "code_pythia410m_step0", "code_pythia410m_step143000",
            "sorted_bow", "sorted_pythia410m_step0", "posword_bow", "posword_pythia410m_step0"]
    for (space, target), sub in med.groupby(level=["space", "target"]):
        t = sub.droplevel(["space", "target"]).unstack()
        say("", f"[{space} | {target}]", t.loc[[f for f in show if f in t.index]].to_string())

    # N@10 是否也跟着词频走：对数尺度上的相关系数
    pts = []
    for (space, target, feat, n), v in med.items():
        if feat.endswith("_bow"):
            fmt = feat[:-4]
            for llm in ["pythia410m_step0", "pythia410m_step143000"]:
                w = med.get((space, target, f"{fmt}_{llm}", n))
                if w is not None:
                    pts.append((fmt, llm, np.log10(v), np.log10(w)))
    p = pd.DataFrame(pts, columns=["fmt", "llm", "bow", "llm_val"])
    for llm, g in p.groupby("llm"):
        say(f"  log10(N@10)：{llm} 和词频的相关系数 r = {np.corrcoef(g.bow, g.llm_val)[0, 1]:.3f}（{len(g)} 个点）")

    say("", "  只看分组的 N@10 最好情况（上帝视角，整个结构池；括号里是 N@10 = 1 的概率）")
    for space in S.TARGETS:
        ys, oh, cnt = S.load_space(space)
        bow = np.load(S.FMT_DIR / space / "code_bow.npy")
        for target, y in ys.items():
            a, pa = oracle_n_at_10(cnt, y)
            b, pb = oracle_n_at_10(bow, y)
            say(f"  {space:8s} {target:16s} 只看计数：{a:6.1f}（{pa:.0%}）   只看 code 词频：{b:6.1f}（{pb:.0%}）"
                f"   池子大小 {len(y)}")
    return df


# ---------------------------------------------------------------- B. 模拟搜索

def search_once(X, y, run, budget, init=10, step=10):
    """一次搜索，返回每测完一批后"找到的最好结构在池子里排第几"。X=None 表示随机搜索。"""
    rng = np.random.default_rng(1000 + run)
    true_rank = np.argsort(np.argsort(-y)) + 1
    tested = list(rng.choice(len(y), size=init, replace=False))
    history = [(len(tested), int(true_rank[tested].min()))]
    while len(tested) < budget:
        rest = np.setdiff1d(np.arange(len(y)), tested)
        if X is None:
            new = rng.choice(rest, size=step, replace=False)
        else:
            model = make_head("rf", run)
            model.set_params(regressor__randomforestregressor__n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = model.fit(X[tested], y[tested]).predict(X[rest])
            new = rest[np.argsort(-pred)[:step]]
        tested += list(new)
        history.append((len(tested), int(true_rank[tested].min())))
    return history


def run_search(args, say, out_dir):
    say("", "===== B. 模拟搜索（随机森林预测器，每次测 10 个；数字 = 找到的最好结构在整个池子里的名次，中位数）=====")
    part_file = out_dir / "search_partial.csv"
    prev = pd.read_csv(part_file) if part_file.exists() else pd.DataFrame(
        columns=["space", "target", "features", "run", "queries", "best_rank"])
    done = set(map(tuple, prev[["space", "target", "features", "run"]].drop_duplicates().values))
    jobs = [(s, t, f) for s in S.TARGETS for t in S.TARGETS[s] for f in SEARCH_FEATS
            if any((s, t, f, r) not in done for r in range(args.runs))]
    bar = Progress(len(jobs), name="搜索", every=120)
    rows = []
    with Parallel(n_jobs=args.jobs) as pool:
        for space in S.TARGETS:
            ys, oh, cnt = S.load_space(space)
            feats = {"onehot": oh, "counts+onehot": np.hstack([cnt, oh]), "random": None}
            for target, y in ys.items():
                for f in SEARCH_FEATS:
                    if (space, target, f) not in jobs:
                        continue
                    X = feats[f] if f in feats else np.load(S.FMT_DIR / space / f"{f}.npy")
                    todo = [r for r in range(args.runs) if (space, target, f, r) not in done]
                    out = pool(delayed(search_once)(X, y, r, args.budget) for r in todo)
                    for r, hist in zip(todo, out):
                        rows += [dict(space=space, target=target, features=f, run=r, queries=q, best_rank=b)
                                 for q, b in hist]
                    pd.concat([prev, pd.DataFrame(rows)]).to_csv(part_file, index=False)
                    bar.update(1, note=f"{space} {target} {f}")
    df = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).astype({"queries": int, "best_rank": int, "run": int})
    df.to_csv(out_dir / "search.csv", index=False)

    marks = sorted({q for q in [50, 100, 200] if q <= args.budget} | {args.budget})
    for (space, target), sub in df.groupby(["space", "target"]):
        t = sub[sub.queries.isin(marks)].groupby(["features", "queries"]).best_rank.median().unstack()
        found = sub[sub.queries == max(marks)].groupby("features").best_rank.apply(lambda s: (s == 1).mean())
        t[f"找到第一的比例@{max(marks)}"] = found
        say("", f"[{space} | {target}]", t.loc[[f for f in SEARCH_FEATS if f in t.index]].to_string())
        # 配对比较：同一个 run（同样的初始 10 个结构），在预算用完时和 one-hot 比名次
        last = sub[sub.queries == max(marks)].pivot(index="run", columns="features", values="best_rank")
        out = []
        for f in SEARCH_FEATS:
            if f != "onehot" and f in last and "onehot" in last:
                d = (last["onehot"] - last[f]).values  # 正数 = f 找到的更好（名次更小）
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                out.append(f"  {f} 比 one-hot 更好的次数 {(d > 0).sum()}/{len(d)}，更差 {(d < 0).sum()}，p={p:.2g}")
        say(*out)
    return df


def plot_search(df, out_dir):
    cols = list(df.groupby(["space", "target"]).groups)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.3 * len(cols), 4.3), squeeze=False)
    for ax, (space, target) in zip(axes[0], cols):
        sub = df[(df.space == space) & (df.target == target)]
        for f in SEARCH_FEATS:
            g = sub[sub.features == f].groupby("queries").best_rank.median()
            if not g.empty:
                ax.plot(g.index, g.values, color=COLOR[f], label=LABEL[f],
                        linewidth=2.5 if f in ("onehot", "code_pythia410m_step0") else 1.5,
                        linestyle=":" if f == "random" else "-")
        ax.set_yscale("log")
        ax.set_title(f"{space} | {target}", fontsize=10)
        ax.set_xlabel("architectures evaluated")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("rank of best found in pool (median, lower = better)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Predictor-guided search (random forest predictor, 10 new architectures per round)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_search.png", dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out", default=str(HERE / "results" / "step6"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    n10_analysis(say)
    df = run_search(args, say, out_dir)
    plot_search(df, out_dir)
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
