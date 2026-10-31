"""第 2 步 + 换搜索空间：检验"数积木"解释的几个预言，并在 TransNAS-Bench-101 上看现象会不会重复。

第 1 步（explain.py）的结论：LLM 特征主要记住了"每种操作各用了几次"（操作计数），
不太在意操作放在哪条边上。如果这是真的，下面几件事应该成立:

A. 2×2 实验（NB201，CIFAR-10）
   文本里的算子名：有意义（conv3x3）/ 匿名（op_D）
   操作所在的边：  原样 / 打乱（每个结构把 6 个操作随机换边，计数不变）
   打乱后，文本描述的其实是"另一个结构"，但标签仍是原结构的精度。
   预言 1：LLM 特征打乱后 τ 几乎不变。对照组 one-hot 打乱后应该明显下降（只剩计数信息）。
   预言 2：算子名匿名后，随机初始化的 Pythia 不受影响（它本来就不懂这些词）。

B. 分行特征（NB201，CIFAR-10）
   x1、x2、x3 三行代码分别取平均再拼起来，保留"哪个操作在哪一行"。
   预言 3：样本多时，分行特征应该能突破计数天花板（平均特征做不到）。
   对照：打乱文本的分行特征仍然应该被天花板卡住。

C. TransNAS-Bench-101 micro（4096 个结构，4 种操作，7 个任务）
   和 NB201 同样的 cell，但任务完全不同（场景分类、拼图、语义分割、自编码器……）。
   看"小样本 LLM 好、样本多被 one-hot 反超、计数能复现小样本优势"是否重复。
   标签用 benchmark 提供的验证集指标（*_valid）。

训练集抽法和 evaluate.py / explain.py 完全一样（同一个种子抽同一批结构），
所以 NB201 上已经算过的组合（one-hot、计数、两个 Pythia 的原始平均特征）直接复用。

运行方式（在 llm_demo 文件夹下）:
    python step2.py                     完整实验（20 个种子）
    python step2.py --seeds 1 --max-n 100 --out results/_smoke   冒烟测试
    python step2.py --only transnas     只跑 C（nb201 同理）
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
from evaluate import HEADS, TRAIN_SIZES
from scipy.stats import wilcoxon

from explain import count_ceiling, probe, run_one
from nb101_text import OPS as NB101_OPS
from nb101_text import encode_nb101, load_nb101
from progress import Progress

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "features"
TRANSNAS_CSV = HERE.parent / "predictor_lab" / "data" / "transnas_micro.csv"

NB_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
TN_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3"]
EDGE_COLS = ["op_0to1", "op_0to2", "op_1to2", "op_0to3", "op_1to3", "op_2to3"]
TN_TASKS = ["class_scene", "class_object", "jigsaw", "room_layout", "segmentsemantic", "normal", "autoencoder"]

PY = {"pythia410m_step0": "Pythia step 0 (random)", "pythia410m_step143000": "Pythia step 143000"}
# NB201 要比较的特征：手工特征 + 两个 Pythia 的 6 种变体
NB_FEATS = ["onehot", "onehot_shuf", "counts", "bow"] + [
    m + v for m in PY for v in ["", "_shuf", "_anon", "_anon_shuf", "_lines", "_shuf_lines"]]
TN_FEATS = ["onehot", "counts", "counts+onehot", "bow", "pythia410m_step0", "pythia410m_step143000", "qwen25_0.5b",
            "pythia410m_step0_lines"]
# counts+wiring：代码文本里几个"数得出来"的量（3 种操作各几次、总连线数、x0 被用了几次、输出拼接了几路），
# 也就是平均特征最容易记住的"词频"信息
NB101_FEATS = ["onehot", "counts", "counts+wiring", "bow", "counts+onehot", "pythia410m_step0", "pythia410m_step143000"]
# TransNAS 只有 4096 个结构，训练集最多抽 1000 个（留 3000 多个测试）
TN_SIZES = [10, 50, 100, 500, 1000]


def seeds_for(n, n_seeds):
    """样本多时各种子的结果很稳（标准差约 0.01），随机森林又特别慢，所以 n ≥ 1000 只跑一半种子。"""
    return n_seeds if n <= 500 else max(1, n_seeds // 2)


# ---------------------------------------------------------------- 数据和特征

def encode(archs, ops):
    """返回 (one-hot, 计数, 每条边的操作编号)。one-hot 的排法和 arch_text.one_hot 一样（边在外层）。"""
    idx = np.array([[ops.index(o) for o in a] for a in archs])
    oh = np.zeros((len(archs), 6, len(ops)), dtype=np.float32)
    oh[np.arange(len(archs))[:, None], np.arange(6), idx] = 1
    return oh.reshape(len(archs), -1), oh.sum(axis=1), idx


def shuffle_ops(arch, index):
    """把 6 条边上的操作随机换位置。必须和 embed.py 里的同名函数完全一样（冒烟测试会核对）。"""
    perm = np.random.default_rng(index).permutation(len(arch))
    return [arch[p] for p in perm]


def load_space(space):
    """返回 (结构列表, {目标名: 标签数组}, 操作表)。"""
    if space == "nb201":
        archs, y = load_dataset("cifar10_acc")
        return archs, {"cifar10_acc": y}, NB_OPS
    if space == "nb101":  # 只用提过特征的那 5000 个结构（features/nb101/pool.npy）
        all_archs, rows = load_nb101()
        pool = np.load(FEATURE_DIR / "nb101" / "pool.npy")
        y = np.array([float(rows[i]["test_acc"]) for i in pool])
        return [all_archs[i] for i in pool], {"test_acc": y}, NB101_OPS
    df = pd.read_csv(TRANSNAS_CSV)
    archs = df[EDGE_COLS].values.tolist()
    return archs, {t: df[f"{t}_valid"].values.astype(float) for t in TN_TASKS}, TN_OPS


def wiring_counts(archs):
    """NB101 代码里能直接"数"出来的连线信息：总连线数、x0 被几个节点用到、输出拼接了几路。"""
    rows = []
    for a in archs:
        inputs = [a[k] for k in sorted(a) if k.startswith("input")]
        rows.append([sum(len(x) for x in inputs), sum(0 in x for x in inputs), len(inputs[-1])])
    return np.array(rows, dtype=np.float32)


def encode_any(space, archs, ops):
    """返回 (one-hot, 计数)。NB101 的 one-hot 是连线（21 位）+ 每个节点的操作（15 位）。"""
    if space == "nb101":
        pairs = [encode_nb101(a) for a in archs]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    oh, cnt, _ = encode(archs, ops)
    return oh, cnt


def load_feature(space, name, archs, ops):
    """按名字拿特征。大文件用到时才读进内存，用完就扔。"""
    if name in ("onehot", "counts", "counts+onehot", "counts+wiring") and space == "nb101":
        oh, cnt = encode_any(space, archs, ops)
        if name == "counts+wiring":
            return np.hstack([cnt, wiring_counts(archs)])
        return {"onehot": oh, "counts": cnt, "counts+onehot": np.hstack([cnt, oh])}[name]
    if name in ("onehot", "counts", "counts+onehot", "onehot_shuf"):
        if name == "onehot_shuf":
            archs = [shuffle_ops(a, i) for i, a in enumerate(archs)]
        oh, cnt, _ = encode(archs, ops)
        return {"onehot": oh, "onehot_shuf": oh, "counts": cnt, "counts+onehot": np.hstack([cnt, oh])}[name]
    folder = FEATURE_DIR / ("" if space == "nb201" else space)
    return np.load(folder / f"{name}.npy")


# ---------------------------------------------------------------- 学习曲线

def est_cost(head, n, dim):
    """粗估训练一次要几秒（单进程），只用来让进度条的剩余时间估得准一点。数值来自实测:
    随机森林 n=2000 时 1024 维约 120 秒、3072 维约 230 秒；岭回归约 10 秒；MLP 2～6 秒。"""
    if head == "ridge":
        return 0.5 + n / 200
    if head == "rf":
        return 0.0073 * n ** 1.3 * (0.05 + dim / 1300)
    return 0.5 + n / 1000 * (1 + dim / 1500)  # mlp


def curves(space, archs, ys, ops, feats, sizes, n_seeds, n_jobs, out_dir):
    partial_file = out_dir / f"curves_{space}_partial.csv"
    prev = pd.read_csv(partial_file) if partial_file.exists() else pd.DataFrame(
        columns=["target", "n_train", "seed", "features", "head", "tau", "n_at_10"])
    old = [prev]
    if space == "nb201":  # 第 1 步和最早的评估里已经算过的组合
        old += [pd.read_csv(HERE / "results" / "results.csv"), pd.read_csv(HERE / "results" / "step1" / "curves.csv")]
    old = pd.concat(old, ignore_index=True)
    old = old[(old.seed < n_seeds) & old.target.isin(ys) & old.features.isin(feats) & old.n_train.isin(sizes)]
    old = old.drop_duplicates(["target", "features", "head", "n_train", "seed"])
    have = old.groupby(["target", "features", "head", "n_train"]).seed.nunique()

    jobs = [(f, t, h, n) for f in feats for t in ys for h in HEADS for n in sizes
            if have.get((t, f, h, n), 0) < seeds_for(n, n_seeds)]
    total = len(feats) * len(ys) * len(HEADS) * len(sizes)
    print(f"\n[{space}] 学习曲线一共 {total} 组，{total - len(jobs)} 组之前算过直接复用，"
          f"这次要算 {len(jobs)} 组（n≤500 每组 {n_seeds} 个种子，n≥1000 每组 {seeds_for(9999, n_seeds)} 个；"
          f"{n_jobs} 个进程）", flush=True)
    if not jobs:
        return old.astype({"tau": float, "n_at_10": float, "n_train": int, "seed": int})

    dims = {}
    for f in dict.fromkeys(j[0] for j in jobs):  # 先只读文件头拿维度，算进度条的总量
        p = FEATURE_DIR / ("" if space == "nb201" else space) / f"{f}.npy"
        dims[f] = np.load(p, mmap_mode="r").shape[1] if p.exists() else 35
    bar = Progress(sum(est_cost(h, n, dims[f]) * seeds_for(n, n_seeds) for f, _, h, n in jobs),
                   name=f"{space}", every=120, units=False)

    rows, t0 = [], time.time()
    with Parallel(n_jobs=n_jobs) as pool:
        for feat in dict.fromkeys(j[0] for j in jobs):
            X = load_feature(space, feat, archs, ops)
            for f, target, head, n in [j for j in jobs if j[0] == feat]:
                k = seeds_for(n, n_seeds)
                out = pool(delayed(run_one)(X, ys[target], n, s, head) for s in range(k))
                rows += [dict(target=target, n_train=n, seed=s, features=feat, head=head, tau=tau, n_at_10=n10)
                         for s, (tau, n10) in enumerate(out)]
                pd.concat([prev, pd.DataFrame(rows)]).to_csv(partial_file, index=False)  # 随时存盘，断了能接着跑
                bar.update(est_cost(head, n, X.shape[1]) * k, note=f"{feat} {target} {head} n={n}")
            # 每算完一种特征，打一行它在随机森林下的结果，方便边跑边看
            done = pd.DataFrame(rows)
            done = done[(done.features == feat) & (done["head"] == "rf")]
            if not done.empty:
                first = list(ys)[0]
                curve = done[done.target == first].groupby("n_train").tau.mean()
                print(f"    ✔ {feat:28s} {first} rf：" + "  ".join(f"n={k}:{v:.3f}" for k, v in curve.items())
                      + f"   （已用 {(time.time() - t0) / 60:.0f} 分钟）", flush=True)
            del X

    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df = df.astype({"tau": float, "n_at_10": float, "n_train": int, "seed": int})
    return df.drop_duplicates(["target", "features", "head", "n_train", "seed"])


# ---------------------------------------------------------------- 汇总

def paired_common(df, target, head, a, b):
    """和 explain.paired 一样，但只比两边都有的种子（大样本时只跑了一半种子）。"""
    out = []
    sub = df[(df.target == target) & (df["head"] == head)]
    for n in sorted(sub.n_train.unique()):
        x = sub[(sub.features == a) & (sub.n_train == n)].set_index("seed").tau
        z = sub[(sub.features == b) & (sub.n_train == n)].set_index("seed").tau
        common = x.index.intersection(z.index)
        if len(common) < 5:
            continue
        d = (x[common] - z[common]).values
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out.append(f"n={n}: {d.mean():+.3f}（{(d > 0).sum()}/{len(d)}，p={p:.2g}）")
    return "；".join(out)


def mean_table(df, target, head, order):
    sub = df[(df.target == target) & (df["head"] == head)]
    table = sub.groupby(["features", "n_train"]).tau.mean().unstack()
    return table.loc[[f for f in order if f in table.index]].round(3).to_string()


def plot_nb201(df, ceiling, out_dir):
    groups = {  # 左：2×2（平均特征）；右：分行特征
        "2x2 (mean pooling)": ["", "_shuf", "_anon", "_anon_shuf"],
        "per-line pooling": ["", "_lines", "_shuf_lines"],
    }
    styles = {"": "-", "_shuf": "--", "_anon": ":", "_anon_shuf": "-.", "_lines": "-", "_shuf_lines": "--"}
    colors = {"pythia410m_step0": "tab:blue", "pythia410m_step143000": "tab:red"}
    fig, axes = plt.subplots(len(groups), len(HEADS), figsize=(16, 9), sharex=True, sharey=True)
    for r, (title, variants) in enumerate(groups.items()):
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            sub = df[(df.target == "cifar10_acc") & (df["head"] == head)]

            def line(feat, **kw):
                g = sub[sub.features == feat].groupby("n_train").tau.mean()
                if not g.empty:
                    ax.plot(g.index, g.values, marker="o", markersize=3, **kw)

            line("onehot", color="black", linewidth=2.5, label="one-hot")
            line("onehot_shuf", color="black", linewidth=1.5, linestyle="--", label="one-hot, shuffled")
            line("counts", color="tab:green", linewidth=2, linestyle="--", label="op counts")
            line("bow", color="tab:brown", linewidth=2.5, linestyle="-.", label="token counts (bag of words)")
            for m, mlabel in PY.items():
                for v in variants:
                    line(m + v, color=colors[m], linestyle=styles[v], linewidth=2.5 if v == "_lines" else 1.5,
                         alpha=0.45 if (r == 1 and v == "") else 1, label=f"{mlabel}{v or ' (mean)'}")
            ax.axhline(ceiling, color="tab:green", linestyle=":", linewidth=1.5, label="ceiling of op counts")
            ax.set_xscale("log")
            ax.set_title(f"{title}  |  head = {head}")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Kendall tau-b (CIFAR-10)")
                ax.legend(fontsize=7)
            if r == len(groups) - 1:
                ax.set_xlabel("number of training architectures")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_nb201_2x2_lines.png", dpi=150)


def plot_transnas(df, ceilings, out_dir, tasks=TN_TASKS, fname="fig_transnas.png",
                  title="TransNAS-Bench-101 micro (4096 archs)"):
    styles = {
        "onehot": dict(color="black", linewidth=2.5, label="one-hot"),
        "counts": dict(color="tab:green", linewidth=2, linestyle="--", label="op counts"),
        "counts+onehot": dict(color="tab:purple", linewidth=2, label="counts + one-hot"),
        "counts+wiring": dict(color="tab:olive", linewidth=2.5, linestyle="-.", label="op counts + wiring counts (NB101)"),
        "pythia410m_step0": dict(color="tab:blue", label="Pythia step 0 (random)"),
        "pythia410m_step143000": dict(color="tab:red", label="Pythia step 143000"),
        "qwen25_0.5b": dict(color="tab:orange", label="Qwen2.5-0.5B"),
        "bow": dict(color="tab:brown", linewidth=2.5, linestyle="-.", label="token counts (bag of words)"),
        "pythia410m_step0_lines": dict(color="tab:blue", linestyle=":", label="Pythia step 0, per-line"),
        "pythia410m_step143000_lines": dict(color="tab:red", linestyle=":", label="Pythia step 143000, per-line"),
    }
    fig, axes = plt.subplots(len(HEADS), len(tasks), figsize=(4 * len(tasks) + 2, 10), sharex=True, squeeze=False)
    for r, head in enumerate(HEADS):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            sub = df[(df.target == task) & (df["head"] == head)]
            for feat, st in styles.items():
                g = sub[sub.features == feat].groupby("n_train").tau.mean()
                if not g.empty:
                    ax.plot(g.index, g.values, marker="o", markersize=3, **st)
            ax.axhline(ceilings[task], color="tab:green", linestyle=":", linewidth=1.5, label="ceiling of op counts")
            ax.set_xscale("log")
            ax.set_title(f"{task} | {head}", fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Kendall tau-b")
            if r == len(HEADS) - 1:
                ax.set_xlabel("n train")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=10, help="同时用几个进程；10 个正好一轮跑完 10 个种子")
    parser.add_argument("--max-n", type=int, default=max(TRAIN_SIZES))
    parser.add_argument("--only", choices=["nb201", "transnas", "nb101"], default=None)
    parser.add_argument("--out", type=str, default=str(HERE / "results" / "step2"))
    parser.add_argument("--no-probe", action="store_true", help="跳过线性探针（探针比较慢，可以之后单独跑）")
    parser.add_argument("--probe-only", action="store_true", help="只跑线性探针，结果写到 probes.txt")
    args = parser.parse_args()
    sizes = [n for n in TRAIN_SIZES if n <= args.max_n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        stem = ("probes" if args.probe_only else "summary") + (f"_{args.only}" if args.only else "")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")  # 边算边写，随时能看

    spaces = [args.only] if args.only else ["nb201", "transnas"]

    # ---------- 先算快的：天花板和探针（几分钟）
    data = {}
    for space in spaces:
        archs, ys, ops = load_space(space)
        oh, cnt = encode_any(space, archs, ops)
        ceilings = {t: count_ceiling(cnt, y) for t, y in ys.items()}
        data[space] = (archs, ys, ops, {t: c for t, (c, _) in ceilings.items()})
        say("", f"===== [{space}] 只靠操作计数的天花板（{len(archs)} 个结构）=====",
            *[f"  {t:16s} τ 上限 = {c:.3f}（{g} 种计数组合）" for t, (c, g) in ceilings.items()])
        bow_file = FEATURE_DIR / ("" if space == "nb201" else space) / "bow.npy"
        if bow_file.exists():  # 词频的天花板：词频完全相同的结构只能给出同一个预测
            bow = np.load(bow_file)
            say(*[f"  词频（bow）{t:12s} τ 上限 = {c:.3f}（{g} 种词频组合）"
                  for t, (c, g) in {t: count_ceiling(bow, y) for t, y in ys.items()}.items()])
        if space == "nb101":  # 全部 42 万个结构上的天花板，看抽 5000 个是否有代表性
            all_archs, rows = load_nb101()
            all_cnt = np.array([encode_nb101(a)[1] for a in all_archs])
            c_all, g_all = count_ceiling(all_cnt, np.array([float(r["test_acc"]) for r in rows]))
            say(f"  （全部 {len(all_archs)} 个结构上：τ 上限 = {c_all:.3f}，{g_all} 种计数组合）")
            all_y = np.array([float(r["test_acc"]) for r in rows])
            c_w, g_w = count_ceiling(np.hstack([all_cnt, wiring_counts(all_archs)]), all_y)
            say(f"  计数 + 连线计数（counts+wiring）在全部结构上：τ 上限 = {c_w:.3f}（{g_w} 种组合）")
            continue  # NB101 不做探针
        edge_ops = encode(archs, ops)[2]

        # 探针：读出的都是"原结构"的计数和每条边的操作。打乱文本的特征读不出原结构的边，才说明打乱生效了
        names = ([m + v for m in PY for v in ["", "_shuf", "_lines", "_shuf_lines"]] if space == "nb201"
                 else [f for f in TN_FEATS if f not in ("onehot", "counts", "counts+onehot")])
        if args.no_probe:
            continue
        say("", f"===== [{space}] 线性探针（读出原结构的信息；edge_acc 瞎猜 = {1 / len(ops):.2f}）=====")
        rows = {}
        for name in names:
            t0 = time.time()
            rows[name] = probe(load_feature(space, name, archs, ops), cnt, edge_ops, oh)
            print(f"    探针 {name} 完成（{(time.time() - t0) / 60:.1f} 分钟）", flush=True)
        say(pd.DataFrame(rows).T.round(3).to_string())

    if args.probe_only:
        return

    # ---------- 学习曲线（主要耗时在这里）
    for space in spaces:
        archs, ys, ops, ceil = data[space]
        feats = {"nb201": NB_FEATS, "transnas": TN_FEATS, "nb101": NB101_FEATS}[space]
        space_sizes = sizes if space == "nb201" else [n for n in TN_SIZES if n <= args.max_n]
        df = curves(space, archs, ys, ops, feats, space_sizes, args.seeds, args.jobs, out_dir)
        df.to_csv(out_dir / f"curves_{space}.csv", index=False)

        say("", f"===== [{space}] 学习曲线：Kendall τ-b 平均值（{args.seeds} 个种子）=====")
        for target in ys:
            for head in HEADS:
                say("", f"[{target} | {head}]   计数天花板 = {ceil[target]:.3f}", mean_table(df, target, head, feats))

        say("", f"===== [{space}] 配对比较：差值（前者赢的次数，Wilcoxon p）=====")
        if space == "nb201":
            pairs = [("onehot_shuf", "onehot"), ("onehot_shuf", "counts"), ("bow", "pythia410m_step0"),
                     ("bow", "counts")]
            for m in PY:
                pairs += [(m + "_shuf", m), (m + "_anon", m), (m + "_anon_shuf", m),
                          (m + "_lines", m), (m + "_lines", "onehot"), (m + "_shuf_lines", m + "_lines")]
        else:
            pairs = [("counts", "pythia410m_step0"), ("counts+wiring", "pythia410m_step0"), ("bow", "pythia410m_step0"), ("pythia410m_step0", "onehot"),
                     ("counts+onehot", "onehot"), ("pythia410m_step143000", "pythia410m_step0"),
                     ("pythia410m_step0_lines", "pythia410m_step0")]
            pairs = [(a, b) for a, b in pairs if a in feats and b in feats]
        for target in ys:
            for head in HEADS:
                say("", f"[{target} | {head}]",
                    *[f"  {a} − {b} : " + paired_common(df, target, head, a, b) for a, b in pairs])

        sub = df[df["head"] == "rf"]
        say("", f"===== [{space}] N@10（随机森林，中位数，越小越好）=====",
            sub.groupby(["target", "features", "n_train"]).n_at_10.median().unstack().to_string())

        if space == "nb201":
            plot_nb201(df, ceil["cifar10_acc"], out_dir)
        elif space == "transnas":
            plot_transnas(df, ceil, out_dir)
        else:
            plot_transnas(df, ceil, out_dir, tasks=list(ys), fname="fig_nb101.png",
                          title="NAS-Bench-101 (5000 random archs)")

    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    main()
