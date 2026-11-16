"""强基线检验：DeepSeek 的"30 个例子 -> 0.60"，是不是真的比认真做的传统方法强？

背景
    之前对照 DeepSeek 的只有 30 个标签的随机森林（0.43），这个对手偏弱。
    这里换成认真设计的几类方法，用和 DeepSeek 完全相同的测试架构、完全相同的 30 个训练例子
    （被留出的操作仍然没见过）：

    rf_onehot / rf_hand     之前的两个随机森林，作为参照
    rf_struct               随机森林 + "结构特征"：4 条输入->输出路径各自是否连通、路径上卷积/跳连/池化/
                            卷积核大小的合计，以及全图各属性合计（用操作的属性，所以对新操作也有定义；
                            思路类似路径编码 / GRAF 这类图特征）
    ridge_struct            岭回归 + 同样的结构特征（标签很少时，线性模型往往最稳）
    gbr_struct              梯度提升树 + 同样的结构特征
    gnn                     小型图神经网络（把 6 条边当节点，沿"上一条边的终点 = 下一条边的起点"传递信息，
                            节点特征是操作属性），用 30 个标签训练
    prior                   零标签的"经验公式"：每条连通路径上各操作的权重之和（卷积3x3=1.0，卷积1x1=0.7，
                            跳连=0.3，池化=0.2，none=路径断开）。权重是跑之前凭常识定的，没有根据结果调整。
                            它代表"不用学习、只靠领域常识"能做到什么——和 LLM 带来的"先验知识"最可比。
    rf_struct_prior / ridge_struct_prior   把 prior 分数也当成一个特征，再用 30 个标签学习修正

判断标准（跑之前写下）
    取上面方法里"10 个种子平均最好的"那个，叫 B*。
    在和 DeepSeek 相同的 3 个种子（0,1,2）上配对比较：差 = DeepSeek(Pro) - B*
        差 >= +0.05   DeepSeek 的优势在强基线面前仍然成立
        差 <= +0.03   没有证明优势（包括 B* 反超），"LLM 少样本预测更强"这个说法站不住
        中间           不确定，需要更多测试架构/种子

用法
    python -I strong_baselines.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_op_holdout import DATA, EDGES, HAND, OPS, OUT, TARGET, arch_features  # noqa: E402
from run_deepseek import K_SHOT, N_TEST  # noqa: E402

SEEDS = list(range(10))          # 前 3 个（0,1,2）与 DeepSeek 的例子集合完全相同
DS_SEEDS = [0, 1, 2]
H = np.array([HAND[o] for o in OPS], dtype=float)        # [有参数, 卷积核, 有输出, 是池化, 是卷积]
# 路径属性用到的 5 列：有参数、卷积核大小、是池化、是卷积、是跳连（恒等）
ATTR = np.stack([H[:, 0], H[:, 1], H[:, 3], H[:, 4], H[:, 2] * (1 - H[:, 0]) * (1 - H[:, 3])], axis=1)
# 边顺序 = EDGES：0:0->1  1:0->2  2:1->2  3:0->3  4:1->3  5:2->3；输入到输出的 4 条路径
PATHS = [[3], [0, 4], [1, 5], [0, 2, 5]]
W_PRIOR = np.array([0.0, 0.3, 0.7, 1.0, 0.2])            # 按 OPS 顺序：none, skip, conv1x1, conv3x3, pool（先验权重）


# ======================================================================================
# 第 1 段：特征
# ======================================================================================
def path_valid(ops, path):
    return (ops[:, path] != 0).all(1).astype(float)      # 路径上没有 none 才连通


def struct_features(op_idx):
    """结构特征：每条路径是否连通 + 路径上的属性合计，再加全图属性合计。"""
    feats = []
    n_valid = np.zeros(len(op_idx))
    tot = np.zeros((len(op_idx), 5))
    for p in PATHS:
        v = path_valid(op_idx, p)
        a = ATTR[op_idx[:, p]].sum(1) * v[:, None]        # (架构数, 5)
        feats += [v[:, None], a]
        n_valid += v
        tot += a
    feats += [n_valid[:, None], tot, ATTR[op_idx].sum(1)]
    return np.hstack(feats)


def prior_score(op_idx):
    """零标签的经验公式：所有连通路径上操作权重之和；一条连通路径都没有则为 -1。"""
    s = np.zeros(len(op_idx))
    any_valid = np.zeros(len(op_idx), dtype=bool)
    for p in PATHS:
        v = path_valid(op_idx, p)
        s += W_PRIOR[op_idx[:, p]].sum(1) * v
        any_valid |= v > 0
    s[~any_valid] = -1.0
    return s


# ======================================================================================
# 第 2 段：小型图神经网络（6 条边当 6 个节点）
# ======================================================================================
def gnn_inputs(op_idx):
    import torch
    x_attr = H[op_idx]                                              # (N,6,5)
    pos = np.broadcast_to(np.eye(6), (len(op_idx), 6, 6))           # 边位置的 one-hot
    return torch.tensor(np.concatenate([x_attr, pos], axis=2), dtype=torch.float32)


def gnn_adj():
    import torch
    head = [1, 2, 2, 3, 3, 3]
    tail = [0, 0, 1, 0, 1, 2]
    A = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            if head[a] == tail[b]:
                A[a, b] = 1.0            # 边 a 的终点就是边 b 的起点：a 的输出流入 b
    return torch.tensor(A, dtype=torch.float32)


def gnn_fit_predict(Xtr, ytr, Xte, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    A = gnn_adj()

    class Net(nn.Module):
        def __init__(s, d=11, h=16):
            super().__init__()
            s.l1, s.l2, s.out = nn.Linear(3 * d, h), nn.Linear(3 * h, h), nn.Linear(2 * h, 1)

        def layer(s, lin, z):
            pred = torch.einsum("ab,Bad->Bbd", A, z)          # 来自上游边的信息
            succ = torch.einsum("ab,Bbd->Bad", A, z)          # 来自下游边的信息
            return torch.relu(lin(torch.cat([z, pred, succ], dim=-1)))

        def forward(s, x):
            z = s.layer(s.l2, s.layer(s.l1, x))
            return s.out(torch.cat([z.mean(1), z.max(1).values], dim=-1)).squeeze(-1)

    mu, sd = ytr.mean(), ytr.std() + 1e-9
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = ((net(Xtr) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(Xte).numpy() * sd + mu


# ======================================================================================
# 第 3 段：主流程
# ======================================================================================
def main():
    df = pd.read_csv(DATA).dropna(subset=[TARGET]).reset_index(drop=True)
    op_idx = df[EDGES].apply(lambda s: s.map(OPS.index)).values
    y = df[TARGET].values
    X_onehot = arch_features(op_idx, np.eye(5))
    X_hand = arch_features(op_idx, H)
    X_struct = struct_features(op_idx)
    prior = prior_score(op_idx)
    X_struct_prior = np.hstack([X_struct, prior[:, None]])
    X_gnn = gnn_inputs(op_idx)

    def rf(X, tr, te, seed):
        return RandomForestRegressor(200, n_jobs=-1, random_state=seed).fit(X[tr], y[tr]).predict(X[te])

    def ridge(X, tr, te, seed):
        sc = StandardScaler().fit(X[tr])
        return RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))

    def gbr(X, tr, te, seed):
        return GradientBoostingRegressor(n_estimators=100, max_depth=2, subsample=0.8,
                                         random_state=seed).fit(X[tr], y[tr]).predict(X[te])

    methods = {
        "rf_onehot": lambda tr, te, s: rf(X_onehot, tr, te, s),
        "rf_hand": lambda tr, te, s: rf(X_hand, tr, te, s),
        "rf_struct": lambda tr, te, s: rf(X_struct, tr, te, s),
        "ridge_struct": lambda tr, te, s: ridge(X_struct, tr, te, s),
        "gbr_struct": lambda tr, te, s: gbr(X_struct, tr, te, s),
        "gnn": lambda tr, te, s: gnn_fit_predict(X_gnn[tr], y[tr], X_gnn[te], s),
        "rf_struct_prior": lambda tr, te, s: rf(X_struct_prior, tr, te, s),
        "ridge_struct_prior": lambda tr, te, s: ridge(X_struct_prior, tr, te, s),
    }

    rows, t0 = [], time.time()
    for held in OPS:
        h = OPS.index(held)
        has_h = (op_idx == h).any(1)
        pool = np.where(~has_h)[0]
        # 与 run_deepseek.py 完全相同的测试架构和例子集合
        test = np.random.default_rng(100 + h).choice(np.where(has_h)[0], N_TEST, replace=False)
        rows.append(dict(held=held, seed=-1, method="prior(0标签)", rho=spearmanr(prior[test], y[test])[0]))
        for seed in SEEDS:
            tr = np.random.default_rng(seed).choice(pool, K_SHOT, replace=False)
            for name, fn in methods.items():
                rows.append(dict(held=held, seed=seed, method=name, rho=spearmanr(fn(tr, test, seed), y[test])[0]))
        print(f"  完成留出 {held}，已用 {time.time() - t0:.0f} 秒", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "strong_baselines_raw.csv", index=False)

    # ---------- 汇总（10 个种子平均）----------
    tab = res.groupby(["method", "held"]).rho.mean().unstack("held")
    tab["平均"] = tab.mean(1)
    tab = tab.sort_values("平均", ascending=False)
    print("\n===== 30 个标签、没见过被留出操作：各方法的 Spearman（10 个种子平均）=====")
    print(tab.round(3).to_string())
    tab.round(4).to_csv(OUT / "strong_baselines_summary.csv")

    # ---------- 和 DeepSeek 配对比较（种子 0,1,2）----------
    cand = [m for m in tab.index if m != "prior(0标签)"]
    best = tab.loc[cand, "平均"].idxmax()
    print(f"\n最强的需要标签的基线 B* = {best}（10 个种子平均 {tab.loc[best, '平均']:.3f}）；"
          f"零标签经验公式 prior = {tab.loc['prior(0标签)', '平均']:.3f}")
    for label, f in [("DeepSeek-v4-pro（关思考）", "deepseek_deepseek-v4-pro-think-off_raw.csv"),
                     ("deepseek-chat（别名）", "deepseek_deepseek-chat_raw.csv")]:
        p = OUT / f
        if not p.exists():
            continue
        ds = pd.read_csv(p)
        ds = ds[(ds.variant == "ds_k30") & ds.seed.isin(DS_SEEDS)]
        base = res[(res.method == best) & res.seed.isin(DS_SEEDS)]
        m = ds.merge(base, on=["held", "seed"], suffixes=("_ds", "_b"))
        d_all = m.rho_ds.mean() - m.rho_b.mean()
        per_op = (m.groupby("held").rho_ds.mean() - m.groupby("held").rho_b.mean()).round(3).to_dict()
        print(f"\n{label}：平均 {m.rho_ds.mean():.3f}，同样 3 个种子上的 B* {m.rho_b.mean():.3f}，差 {d_all:+.3f}")
        print(f"   每个操作的差：{per_op}")
        if "pro" in label.lower():
            verdict = ("DeepSeek 的优势在强基线面前仍然成立" if d_all >= 0.05 else
                       "没有证明优势：'LLM 少样本预测更强'站不住" if d_all <= 0.03 else "不确定：需要更多测试架构/种子")
            print(f"   按预先写好的标准（>=+0.05 成立，<=+0.03 不成立）：{verdict}")


if __name__ == "__main__":
    main()
