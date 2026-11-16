"""补充实验：Qwen 的 LLM 特征和"Qwen 自己的分词器"数出来的词频比，是否也一致？

为什么要补：规律 2、3 里 Qwen 的吻合程度（0.86），是拿 Qwen 的特征和 Pythia 分词器数出来的词频比的，
两边不是同一套切词方式，这个比较不成立。这里改用 Qwen 自己的分词器重新数词频，其他条件完全不变:
    结构池   features/formats/<空间>/pool.npy（和原实验同一批结构、同一个顺序）
    文本     formats.py 的 5 种写法（和送进 Qwen 的文本一模一样）
    训练划分 同样的种子抽同样的训练集（n ≤ 500 时 20 个种子，n = 1000 时 10 个）
    回归头   岭回归、随机森林
Qwen 的 LLM 特征的学习曲线直接用原实验的结果（results_repaired/step3），因为划分和种子完全相同。

比较方式：只在"同一个样本数"内比较（每个样本数 25 个点 = 5 种写法 × 5 个空间-任务），
不把不同样本数的点混在一起算相关系数，因为样本数本身会同时抬高两边的 τ。

运行方式（在 llm_demo 文件夹下）:
    python step3b_qwen_bow.py --seeds 2 --max-n 50 --out results_v3/_smoke_qwen_bow   冒烟测试
    python step3b_qwen_bow.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import step3_formats as S
from artifacts import atomic_npy, feature_cached, feature_context, save_feature
from embed import load_archs
from experiment import atomic_csv, run_curves
from formats import FORMATS, to_text
from make_bow import TOKENIZER as PYTHIA_TOKENIZER, bow
from models import MODEL_DIR
from step2 import seeds_for

HERE = Path(__file__).parent
QWEN_TOKENIZER = MODEL_DIR / "qwen25_0.5b"
SPACES = list(S.TARGETS)  # nb201、transnas、nb101，各自的目标和原实验一样
LLM = "qwen25_0.5b"


def make_features(space, out_dir, tokenizer):
    """用 Qwen 分词器给 5 种写法各数一份词频，存成 <写法>_qwenbow.npy。"""
    pool = np.load(S.FMT_DIR / space / "pool.npy")
    archs = load_archs(space)
    folder = out_dir / space
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "pool.npy").exists():
        if not np.array_equal(np.load(folder / "pool.npy"), pool):
            raise ValueError(f"{folder} 里已有的结构池和原实验不同，请换一个输出文件夹")
    else:
        atomic_npy(folder / "pool.npy", pool)
    for fmt in FORMATS:
        texts = [to_text(space, archs[i], fmt) for i in pool]
        path = folder / f"{fmt}_qwenbow.npy"
        context = feature_context(texts, pool, QWEN_TOKENIZER, "bow-qwen")
        if feature_cached(path, context, len(pool)):
            continue
        X, words = bow(texts, tokenizer)
        save_feature(path, X, context)
        print(f"[Qwen 词频] {space:8s} {fmt:8s} {X.shape[1]:3d} 种会变的 token："
              f"{' '.join(repr(w) for w in words[:15])}{' ……' if len(words) > 15 else ''}", flush=True)
    return folder


def compare(new_curves, out_dir, say):
    """同一样本数内：Qwen 特征 vs Qwen 词频，以及（作对照）vs Pythia 词频。"""
    rows = []
    for space in SPACES:
        old = pd.read_csv(HERE / "results_repaired" / "step3" / f"curves_{space}.csv")
        new = new_curves[space]
        both = pd.concat([old, new], ignore_index=True)
        for target in S.TARGETS[space]:
            for head in S.HEADS:
                for fmt in FORMATS:
                    for n in sorted(new.n_train.unique()):
                        cols = {"llm": f"{fmt}_{LLM}", "qwen_bow": f"{fmt}_qwenbow", "pythia_bow": f"{fmt}_bow"}
                        sub = both[(both.target == target) & (both["head"] == head) & (both.n_train == n)
                                   & both.features.isin(cols.values())]
                        wide = sub.pivot(index="seed", columns="features", values="tau").reindex(columns=list(cols.values()))
                        wide = wide.dropna()  # 三种特征都有有效 τ 的种子才算，保证在同一批训练划分上比较
                        rows.append(dict(space=space, target=target, head=head, format=fmt, n_train=n,
                                         seeds=len(wide), **{k: wide[v].mean() for k, v in cols.items()}))
    pts = pd.DataFrame(rows)
    atomic_csv(pts, out_dir / "points.csv")

    out = []
    for (head, n), g in pts.groupby(["head", "n_train"]):
        g = g.dropna()
        row = dict(head=head, n_train=n, points=len(g))
        for base in ["qwen_bow", "pythia_bow"]:
            row[f"r_{base}"] = g.llm.corr(g[base])
            row[f"gap_{base}"] = (g.llm - g[base]).mean()       # 平均偏差：LLM − 词频
            row[f"absgap_{base}"] = (g.llm - g[base]).abs().mean()
        out.append(row)
    table = pd.DataFrame(out)
    atomic_csv(table, out_dir / "agreement_by_n.csv")
    say("", "===== Qwen 特征和词频的吻合程度（同一样本数内；每个样本数 25 个点）=====",
        "  r_* = 相关系数；gap_* = 平均偏差（Qwen 特征 τ − 词频 τ）；absgap_* = 平均差距",
        table.round(3).to_string(index=False))

    # 两种分词器数出来的词频本身差多少
    diff = (pts.qwen_bow - pts.pythia_bow).dropna()
    say("", f"Qwen 词频 τ − Pythia 词频 τ：平均 {diff.mean():+.3f}，最大绝对差 {diff.abs().max():.3f}（{len(diff)} 个点）")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=max(S.SIZES))
    parser.add_argument("--features-out", default=str(HERE / "features_generated_v3" / "qwen_bow"))
    parser.add_argument("--out", default=str(HERE / "results_v3" / "qwen_bow"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_root = Path(args.features_out)
    sizes = [n for n in S.SIZES if n <= args.max_n]
    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    say(f"Qwen 分词器：{QWEN_TOKENIZER}；对照用的 Pythia 分词器：{PYTHIA_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    curves = {}
    for space in SPACES:
        folder = make_features(space, feat_root, tokenizer)
        ys, _, _ = S.load_space(space)  # 标签和原实验同一套（会先核对原特征文件夹没被改动）
        feats = [f"{fmt}_qwenbow" for fmt in FORMATS]
        df = run_curves(space, ys, feats, lambda f: np.load(folder / f"{f}.npy"),
                        sizes, args.seeds, args.jobs, out_dir, S.HEADS, seeds_for)
        atomic_csv(df, out_dir / f"curves_{space}.csv")
        curves[space] = df
    compare(curves, out_dir, say)
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
