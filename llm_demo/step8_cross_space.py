"""第 8 步（初步）：跨搜索空间迁移。检验动机里的问题：LLM 编码能不能跨空间通用？

设置参照 CDP（Liu 等，NeurIPS 2022）：它也用 NB101 和 NB201 作为源空间。
两个空间都是 CIFAR-10，所以标签含义一致，只是架构空间不同。

两个方向：NB101 → NB201、NB201 → NB101。三种设置：
    zero      只用源空间 N_SRC 个有标签结构训练，直接给目标空间全部结构排序
    target    只用目标空间 n 个有标签结构训练（不迁移，作为参照）
    combined  源空间 N_SRC 个 + 目标空间 n 个一起训练
前两种之外的结构都作为目标空间的测试集，指标是 Kendall τ。

四种特征（都能跨空间使用，one-hot 不能，所以不在比较之列）:
    random_pythia   随机初始化 Pythia-410M 的平均特征（主实验已提取，code 写法）
    trained_pythia  训练完成 Pythia-410M 的平均特征
    token_counts    两个空间合并词表后重新数的词频（同一个 Pythia 分词器）
    shared_counts   两个空间共有的操作计数：conv3x3、conv1x1

预先写下的预言（按 Q1 的结论）：如果 LLM 编码在空间内基本等于词频，
那么跨空间时它能迁移的只有两边共有的词，所以 LLM 特征不会明显好于 token_counts。

两个基准的精度尺度不同（训练轮数等不同），所以每个训练集的标签先换成 0～1 的百分位排名再训练。
回归头用主实验同一个随机森林（evaluate.make_head，100 棵树）。

运行方式（在 llm_demo 文件夹下）:
    python step8_cross_space.py                        完整实验
    python step8_cross_space.py --seeds 1 --out results_v3/_smoke_cross   冒烟测试
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, rankdata
from transformers import AutoTokenizer

from arch_text import arch_to_code
from evaluate import make_head
from make_bow import TOKENIZER
from nb101_text import encode_nb101, nb101_to_code
from progress import Progress
from step2 import FEATURE_DIR, NB_OPS, encode, load_space

HERE = Path(__file__).parent
N_SRC = 1000            # 源空间训练用多少个结构
TARGET_NS = [10, 50]    # 目标空间给多少个标签
SPACES = ["nb101", "nb201"]
FEATS = ["random_pythia", "trained_pythia", "token_counts", "shared_counts"]


def pct(y):
    """标签换成 0～1 的百分位排名，消除两个基准的尺度差异。"""
    return (rankdata(y) - 1) / max(1, len(y) - 1)


def build_features():
    """返回 {空间: {特征名: 矩阵}} 和 {空间: 标签}。两个空间同一特征的维度必须一样。"""
    data, labels, texts = {}, {}, {}
    for sp in SPACES:
        archs, ys, ops = load_space(sp)
        labels[sp] = next(iter(ys.values()))
        folder = FEATURE_DIR / ("" if sp == "nb201" else sp)
        f = {"random_pythia": np.load(folder / "pythia410m_step0.npy"),
             "trained_pythia": np.load(folder / "pythia410m_step143000.npy")}
        if sp == "nb201":
            _, cnt, _ = encode(archs, ops)
            f["shared_counts"] = cnt[:, [NB_OPS.index("nor_conv_3x3"), NB_OPS.index("nor_conv_1x1")]]
            texts[sp] = [arch_to_code(a) for a in archs]
        else:
            cnt = np.array([encode_nb101(a)[1] for a in archs])   # 顺序：conv3x3, conv1x1, maxpool
            f["shared_counts"] = cnt[:, [0, 1]]
            texts[sp] = [nb101_to_code(a) for a in archs]
        data[sp] = f
        for name, X in f.items():
            assert len(X) == len(labels[sp]), (sp, name, X.shape, len(labels[sp]))

    # 合并词表的词频：两个空间所有出现过的 token 放进同一张表
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ids = {sp: [tok(t)["input_ids"] for t in texts[sp]] for sp in SPACES}
    vocab = sorted({i for sp in SPACES for seq in ids[sp] for i in seq})
    col = {v: k for k, v in enumerate(vocab)}
    for sp in SPACES:
        X = np.zeros((len(ids[sp]), len(vocab)), dtype=np.float32)
        for r, seq in enumerate(ids[sp]):
            for i in seq:
                X[r, col[i]] += 1
        data[sp]["token_counts"] = X
    shared = sum(1 for v in vocab if (data["nb101"]["token_counts"][:, col[v]].any()
                                      and data["nb201"]["token_counts"][:, col[v]].any()))
    for name in FEATS:
        assert data["nb101"][name].shape[1] == data["nb201"][name].shape[1], name
    return data, labels, (len(vocab), shared)


def run_one(Xs, ys, Xt, yt, setting, n_tgt, seed):
    """一次训练 + 打分。返回目标空间测试集上的 Kendall τ。"""
    rng = np.random.default_rng(seed)
    src = rng.choice(len(ys), size=N_SRC, replace=False)
    perm = rng.permutation(len(yt))
    tr_t, te_t = (perm[:n_tgt], perm[n_tgt:]) if setting != "zero" else (perm[:0], perm)
    if setting == "zero":
        X, y = Xs[src], pct(ys[src])
    elif setting == "target":
        X, y = Xt[tr_t], pct(yt[tr_t])
    else:
        X = np.vstack([Xs[src], Xt[tr_t]])
        y = np.concatenate([pct(ys[src]), pct(yt[tr_t])])
    model = make_head("rf", seed)
    model.set_params(regressor__randomforestregressor__n_jobs=1)
    pred = model.fit(X, y).predict(Xt[te_t])
    return float(kendalltau(pred, yt[te_t]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=str(HERE / "results_v3" / "cross_space"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data, labels, (nvocab, nshared) = build_features()
    print(f"特征准备好了（{time.time() - t0:.0f} 秒）。合并词表 {nvocab} 个 token，两个空间都出现的 {nshared} 个。",
          flush=True)

    jobs = []
    for src, tgt in [("nb101", "nb201"), ("nb201", "nb101")]:
        for feat in FEATS:
            for setting, ns in [("zero", [0]), ("target", TARGET_NS), ("combined", TARGET_NS)]:
                for n in ns:
                    for seed in range(args.seeds):
                        jobs.append((src, tgt, feat, setting, n, seed))
    bar = Progress(total=len(jobs), name="cross_space", every=90)

    def task(j):
        src, tgt, feat, setting, n, seed = j
        tau = run_one(data[src][feat], labels[src], data[tgt][feat], labels[tgt], setting, n, seed)
        return dict(source=src, target=tgt, features=feat, setting=setting, n_target=n, seed=seed, tau=tau)

    rows = []
    with Parallel(n_jobs=args.jobs, return_as="generator_unordered") as pool:
        for r in pool(delayed(task)(j) for j in jobs):
            rows.append(r)
            bar.update(1, note=f"{r['source']}→{r['target']} {r['features']} {r['setting']}")
    df = pd.DataFrame(rows).sort_values(["source", "features", "setting", "n_target", "seed"])
    df.to_csv(out / "cross_space.csv", index=False)

    summ = (df.groupby(["source", "target", "setting", "n_target", "features"]).tau
              .agg(["mean", "std", "count"]).round(3).reset_index())
    summ.to_csv(out / "summary.csv", index=False)
    lines = [f"合并词表 {nvocab} 个 token，两个空间共有 {nshared} 个；源空间训练 {N_SRC} 个；种子 {args.seeds} 个。", ""]
    for (s, t), g in summ.groupby(["source", "target"]):
        lines.append(f"===== {s} → {t}（Kendall τ 均值 ± 标准差）=====")
        tab = g.assign(v=g["mean"].map("{:.3f}".format) + " ± " + g["std"].map("{:.3f}".format))
        tab = tab.pivot_table(index=["setting", "n_target"], columns="features", values="v", aggfunc="first")
        lines.append(tab[FEATS].to_string())
        lines.append("")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟。结果在 {out}", flush=True)


if __name__ == "__main__":
    main()
