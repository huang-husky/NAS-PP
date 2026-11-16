"""留一操作检验：预测器没见过的"新操作"，LLM 对操作的文字描述能不能帮上忙？

问题（一句话）
    NAS-Bench-201 每条边有 5 种操作。假装其中 1 种"训练时从没见过"（比如 3x3 卷积），
    只用不含它的架构训练预测器，再去给含它的架构排名。
    one-hot 编号对没见过的操作毫无办法；文字描述（"3x3 convolution ..."）
    却可能让预测器"猜到"它和哪些见过的操作相似。这个脚本检验这一点。

比较的几种"操作表示"（每条边用一个向量表示它的操作，6 条边拼起来喂给回归器）
    onehot      编号。没见过的操作，对应那一列在训练时全是 0。
    handcrafted 手写的 5 个属性（有无参数、卷积核大小等）——"人来写先验"的对照，
                放在代码里，在看到任何结果之前就定好了，不会事后调。
    random      每个操作一个随机向量——对照：只要"有个稠密向量"就能有帮助吗？
    LLM 嵌入    把操作的描述送进冻结的 LLM，取最后一层隐藏状态的平均，
                再降到 3 维（PCA 只用"见过的 4 个操作"来拟合，没见过的那个只做变换）。
                包括：qwen25 / qwen25coder / pythia 预训练好的 / pythia 随机初始化(step0)，
                描述有两种写法：raw（原始名字 nor_conv_3x3）和 nat（一句自然语言）。
    ceiling     上限参考：训练集是随机抽的，含有那个操作（等于"见过"）。

放弃条件（在跑之前写下，跑完照此判断）
    以"留一操作后的平均 Spearman"为准，所有被留出的操作取平均：
    1. 最好的预训练 LLM 嵌入 比 onehot 高不到 0.05      -> 没用，放弃
    2. 预训练 LLM 嵌入 比 random 或 pythia 随机初始化 高不到 0.03
                                                         -> 只是"有向量"在起作用，不是语义，放弃
    3. 预训练 LLM 嵌入 比 handcrafted 低超过 0.02        -> 不如手写属性，价值很弱

用法
    python run_op_holdout.py                 全部跑（5 个操作 x 5 个随机种子 x 2 个回归器）
    python run_op_holdout.py --quick         冒烟测试：1 个操作、1 个种子
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # MLP 常见的"没收敛"警告，对这个检验不重要

HERE = Path(__file__).parent
DATA = HERE.parent / "predictor_lab" / "data" / "nasbench201.csv"
MODEL_DIR = HERE.parent / "llm_demo" / "models"   # 已经下载好的模型
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TARGET = "cifar10_acc"      # 预测哪个数据集上的准确率
N_TRAIN = 500               # 每次给预测器多少个带标签的架构
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
EDGES = ["op_0to1", "op_0to2", "op_1to2", "op_0to3", "op_1to3", "op_2to3"]

# ---- 操作的文字描述（跑之前定好，不根据结果改）-------------------------------------
DESC_RAW = {o: o for o in OPS}
DESC_NAT = {
    "none": "zero operation: the edge outputs nothing, so the connection is removed",
    "skip_connect": "skip connection: the identity mapping, the input is passed to the output unchanged",
    "nor_conv_1x1": "1x1 convolution: a pointwise convolution with batch normalization and ReLU",
    "nor_conv_3x3": "3x3 convolution: a convolution with a 3x3 kernel, batch normalization and ReLU",
    "avg_pool_3x3": "3x3 average pooling: averages the values in a 3x3 window, no learnable parameters",
}
# ---- 手写属性：[有可学习参数, 卷积核大小, 有输出, 是池化, 是卷积] --------------------
HAND = {
    "none":         [0, 0, 0, 0, 0],
    "skip_connect": [0, 0, 1, 0, 0],
    "nor_conv_1x1": [1, 1, 1, 0, 1],
    "nor_conv_3x3": [1, 3, 1, 0, 1],
    "avg_pool_3x3": [0, 3, 1, 1, 0],
}
# 参与比较的 LLM：(名字, 模型文件夹, 描述写法)
LLM_SPECS = [
    ("qwen25_nat", "qwen25_0.5b", "nat"),
    ("qwen25_raw", "qwen25_0.5b", "raw"),
    ("qwen25coder_nat", "qwen25coder_0.5b", "nat"),
    ("pythia_pre_nat", "pythia410m_step143000", "nat"),
    ("pythia_rand_nat", "pythia410m_step0", "nat"),   # 随机初始化：没有任何预训练知识
]


# ======================================================================================
# 第 1 段：把每个操作的描述变成向量（只算一次，存到缓存）
# ======================================================================================
def embed_texts(model_name, texts):
    """用冻结的模型，把几句话各变成一个向量（对所有 token 的最后一层隐藏状态取平均）。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = MODEL_DIR / model_name
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()
    vecs = []
    with torch.no_grad():
        for t in texts:                              # 一句一句来，不需要填充符
            ids = tok(t, return_tensors="pt")
            h = model(**ids).last_hidden_state[0]    # (token 数, 隐藏维度)
            vecs.append(h.mean(0).numpy())
    return np.stack(vecs)


def load_op_embeddings():
    """返回 {编码名: (5, 维度) 数组}，行顺序同 OPS。有缓存就直接读。"""
    cache = OUT / "op_embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    out = {}
    for name, model_name, style in LLM_SPECS:
        desc = DESC_NAT if style == "nat" else DESC_RAW
        t0 = time.time()
        out[name] = embed_texts(model_name, [desc[o] for o in OPS])
        print(f"  嵌入 {name:16s} 完成，{out[name].shape}，用时 {time.time() - t0:.0f} 秒", flush=True)
    np.savez(cache, **out)
    return out


# ======================================================================================
# 第 2 段：把"每个操作的向量"拼成"每个架构的特征"
# ======================================================================================
def arch_features(op_idx, op_vec):
    """op_idx: (架构数, 6) 每条边的操作编号；op_vec: (5, d) 每个操作的向量。
    返回 (架构数, 6*d)：6 条边的向量首尾拼接。"""
    return op_vec[op_idx].reshape(len(op_idx), -1)


def make_op_vectors(held, seed, emb):
    """对"被留出的操作 held"，造出所有编码各自的 (5, d) 向量表。"""
    seen = [i for i, o in enumerate(OPS) if o != held]
    vecs = {}
    # one-hot：5 维；没见过的操作那一列训练时全是 0
    vecs["onehot"] = np.eye(5)
    vecs["handcrafted"] = np.array([HAND[o] for o in OPS], dtype=float)
    # random：每个操作一个随机 3 维向量（每个种子不同）
    vecs["random"] = np.random.default_rng(1000 + seed).normal(size=(5, 3))
    # LLM：PCA 只用见过的 4 个操作拟合 -> 3 维，再变换全部 5 个
    for name, e in emb.items():
        pca = PCA(n_components=3).fit(e[seen])
        vecs[name] = pca.transform(e)
    return vecs


# ======================================================================================
# 第 3 段：回归器与评估
# ======================================================================================
def fit_predict(kind, Xtr, ytr, Xte, seed):
    if kind == "rf":
        m = RandomForestRegressor(200, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        return m.predict(Xte)
    # MLP：特征和目标都标准化，否则小样本下几乎学不动
    sx = StandardScaler().fit(Xtr)
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, random_state=seed).fit(sx.transform(Xtr), (ytr - mu) / sd)
    return m.predict(sx.transform(Xte)) * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟测试：1 个操作、1 个种子")
    args = ap.parse_args()

    df = pd.read_csv(DATA).dropna(subset=[TARGET]).reset_index(drop=True)
    op_idx = df[EDGES].apply(lambda s: s.map(OPS.index)).values        # (架构数, 6)
    y = df[TARGET].values
    print(f"数据：{len(df)} 个架构，目标 {TARGET}，每次训练 {N_TRAIN} 个标签", flush=True)

    print("准备操作描述的嵌入……", flush=True)
    emb = load_op_embeddings()

    held_ops = ["nor_conv_3x3"] if args.quick else OPS
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    kinds = ["rf", "mlp"]
    rows = []
    total = len(held_ops) * len(seeds)
    done, t0 = 0, time.time()

    for held in held_ops:
        h = OPS.index(held)
        has_h = (op_idx == h).any(1)                  # 含有被留出操作的架构
        train_pool = np.where(~has_h)[0]              # 训练只能从"不含它"的里面抽
        test = np.where(has_h)[0]                     # 测试：全部含有它的架构
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tr = rng.choice(train_pool, N_TRAIN, replace=False)
            vecs = make_op_vectors(held, seed, emb)
            for enc, ov in vecs.items():
                X = arch_features(op_idx, ov)
                for kind in kinds:
                    p = fit_predict(kind, X[tr], y[tr], X[test], seed)
                    rows.append(dict(held=held, seed=seed, enc=enc, reg=kind,
                                     rho=spearmanr(p, y[test])[0]))
            # 上限参考：训练集从所有架构里随机抽（见过被留出的操作），用 one-hot
            tr_all = rng.choice(len(df), N_TRAIN, replace=False)
            X = arch_features(op_idx, np.eye(5))
            for kind in kinds:
                p = fit_predict(kind, X[tr_all], y[tr_all], X[test], seed)
                rows.append(dict(held=held, seed=seed, enc="ceiling", reg=kind,
                                 rho=spearmanr(p, y[test])[0]))
            done += 1
            el = time.time() - t0
            bar = "#" * int(20 * done / total) + "-" * (20 - int(20 * done / total))
            print(f"[{bar}] {done}/{total}  已用 {el/60:.1f} 分钟，"
                  f"预计剩余 {el / done * (total - done) / 60:.1f} 分钟  （刚完成：留出 {held}，种子 {seed}）",
                  flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / ("raw_quick.csv" if args.quick else "raw.csv"), index=False)

    # ---------------- 汇总：先对种子平均，再对被留出的操作平均 ----------------
    order = ["onehot", "random", "pythia_rand_nat", "handcrafted",
             "qwen25_raw", "qwen25_nat", "qwen25coder_nat", "pythia_pre_nat", "ceiling"]
    for kind in kinds:
        sub = res[res.reg == kind]
        per_op = sub.groupby(["enc", "held"]).rho.mean().unstack("held")
        per_op["平均"] = per_op.mean(1)
        print(f"\n===== 回归器 {kind}：留出操作后的 Spearman（越高越好）=====")
        print(per_op.loc[[e for e in order if e in per_op.index]].round(3).to_string(), flush=True)
        per_op.round(4).to_csv(OUT / f"summary_{kind}{'_quick' if args.quick else ''}.csv")

    # ---------------- 按预先写好的放弃条件打分 ----------------
    print("\n===== 放弃条件检查（以两种回归器各自的'平均'列为准）=====")
    for kind in kinds:
        m = res[res.reg == kind].groupby("enc").rho.mean()
        llm = m[["qwen25_raw", "qwen25_nat", "qwen25coder_nat", "pythia_pre_nat"]]
        best = llm.idxmax()
        c1 = llm.max() - m["onehot"]
        c2 = llm.max() - max(m["random"], m["pythia_rand_nat"])
        c3 = llm.max() - m["handcrafted"]
        print(f"[{kind}] 最好的 LLM = {best} ({llm.max():.3f})")
        print(f"   1. 比 onehot 高 {c1:+.3f}   (要求 >= +0.05)  {'通过' if c1 >= 0.05 else '不通过'}")
        print(f"   2. 比 random/随机初始化 高 {c2:+.3f}   (要求 >= +0.03)  {'通过' if c2 >= 0.03 else '不通过'}")
        print(f"   3. 比 handcrafted 高 {c3:+.3f}   (要求 >= -0.02)  {'通过' if c3 >= -0.02 else '不通过'}")
    json.dump({"target": TARGET, "n_train": N_TRAIN, "seeds": seeds}, open(OUT / "config.json", "w"))


if __name__ == "__main__":
    sys.exit(main())
