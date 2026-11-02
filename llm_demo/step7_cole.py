"""检验：COLE（Somu 等，GECCO 2026）为什么没有出现我们说的上限？

COLE 用冻结的 CodeLlama-7B 读 NB201 结构的 PyTorch 代码，N=3516 时 τ 到了 0.797，
远高于我们算的"计数上限"0.63。我们的规律 2 说"上限由写法的词频决定"，所以预言：
COLE 的写法把位置信息写进了词频（每条边一个名字 self.op_0_2，跳连写成 + node_1），
它的词频上限应该高得多；而且在它的写法下，LLM 特征仍然 ≈ 它的词频。

A. 词频上限（全部结构，不用大模型，几分钟）
   NB201 全部 15625 个（3 个数据集）、TransNAS 全部 4096 个（7 个任务）：
   我们的 code 写法 vs COLE 写法，各有几组、上限多少。
B. 学习曲线（结构池和规律 2 相同：NB201 5000 个，TransNAS 4096 个、3 个任务）
   one-hot、code 词频、code 随机/训练完成 Pythia、cole 词频、cole 随机/训练完成 Pythia；
   回归头：岭回归、随机森林、MLP。
   cole 的 Pythia 特征要先用 run_formats.py 提取：
       python run_formats.py --spaces nb201,transnas --formats cole

运行方式（在 llm_demo 文件夹下）:
    python step7_cole.py --ceiling-only     只做 A
    python step7_cole.py                    A + B
"""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import step3_formats as S
from embed import load_archs
from explain import count_ceiling
from formats import cell_text
from make_bow import TOKENIZER, bow
from nb201 import load_dataset
from step2 import TN_TASKS, TRANSNAS_CSV

HERE = Path(__file__).parent
FEATS = ["onehot", "code_bow", "code_pythia410m_step0", "code_pythia410m_step143000",
         "cole_bow", "cole_pythia410m_step0", "cole_pythia410m_step143000"]
TARGETS = {"nb201": ["cifar10_acc"], "transnas": ["class_scene", "segmentsemantic", "autoencoder"]}


def ceilings(say):
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    say("===== A. 词频上限（全部结构；只知道词频时，同一组的结构都预测为组内真实平均值）=====")
    for space in ["nb201", "transnas"]:
        archs = load_archs(space)
        if space == "nb201":
            ys = {t: load_dataset(t)[1] for t in ["cifar10_acc", "cifar100_acc", "imagenet16_acc"]}
        else:
            df = pd.read_csv(TRANSNAS_CSV)
            ys = {t: df[f"{t}_valid"].values.astype(float) for t in TN_TASKS}
        X = {}
        for fmt in ["code", "cole"]:
            X[fmt], words = bow([cell_text(a, fmt) for a in archs], tok)
            say(f"[{space}] {fmt:5s} 写法：{X[fmt].shape[1]} 种会变的 token：{' '.join(repr(w) for w in words)}")
        for t, y in ys.items():
            (a, ga), (b, gb) = count_ceiling(X["code"], y), count_ceiling(X["cole"], y)
            say(f"  {space:8s} {t:16s} code 写法：上限 {a:.3f}（{ga} 组）   cole 写法：上限 {b:.3f}（{gb} 组）"
                f"   结构数 {len(y)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceiling-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out", default=str(HERE / "results" / "step7"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(*new):
        lines.extend(new)
        print("\n".join(new), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    ceilings(say)
    if args.ceiling_only:
        return

    # ---- B. 学习曲线：借用规律 2 的评估代码，只换特征列表、回归头和目标
    import pandas as pd
    S.FEATS, S.HEADS, S.TARGETS = FEATS, ["ridge", "rf", "mlp"], TARGETS
    sizes = [10, 50, 100, 500, 1000, 2000]
    say("", "===== B. 学习曲线（Kendall τ，种子平均）=====")
    for space in TARGETS:
        ys, oh, _ = S.load_space(space)
        df = S.curves(space, ys, oh, sizes, args.seeds, args.jobs, out_dir)
        df.to_csv(out_dir / f"curves_{space}.csv", index=False)
        for (t, h), sub in df.groupby(["target", "head"]):
            tab = sub.groupby(["features", "n_train"]).tau.mean().unstack()
            say("", f"[{space} | {t} | {h}]", tab.loc[[f for f in FEATS if f in tab.index]].round(3).to_string())
        # 规律 2 的检验：cole 写法下，LLM 和词频差多少
        for (t, h), sub in df.groupby(["target", "head"]):
            m = sub.groupby(["features", "n_train"]).tau.mean().unstack()
            for llm in ["pythia410m_step0", "pythia410m_step143000"]:
                if f"cole_{llm}" in m.index and "cole_bow" in m.index:
                    d = (m.loc[f"cole_{llm}"] - m.loc["cole_bow"]).round(3).to_dict()
                    say(f"  {space} {t} {h}: cole_{llm} − cole_bow = {d}")
    pd.concat([pd.read_csv(out_dir / f"curves_{s}.csv").assign(space=s) for s in TARGETS]).to_csv(
        out_dir / "curves_all.csv", index=False)
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
