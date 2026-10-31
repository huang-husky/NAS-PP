"""第 3 步：用 PyTorch 写一个神经网络预测器，比较两种损失函数。

对应 survey 第 9 页:
    公式 (4) MSE 损失           —— 逐个结构比较"预测精度"和"真实精度"差多少
    公式 (6) 成对排序损失        —— 只比较两个结构"谁高谁低"有没有排对
并和第 2 步的随机森林放在一起比。

为了比较公平，每种方法都用 5 个不同的随机种子各跑一次（每次抽到的 200 个训练结构不同），
报告 Kendall's Tau 的平均值和标准差。

运行方式（在 predictor_lab 文件夹下）:
    python step3_mlp_ranking.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

from nb201 import OPS, OUTPUT_DIR, load_dataset

N_TRAIN = 200  # 假装只训练得起 200 个结构
SEEDS = [0, 1, 2, 3, 4]  # 每种方法重复 5 次
EPOCHS = 500  # 神经网络在训练集上学多少轮
LR = 1e-3  # 学习率：每一步参数改动的幅度
MARGIN = 0.1  # 排序损失里的间隔 m


# ----------------------------------------------------------------------
# 编码：和第 2 步完全一样，6 条边 x 5 种操作 = 30 维的 0/1 向量
# ----------------------------------------------------------------------
def encode(arch):
    vec = np.zeros(30, dtype=np.float32)  # PyTorch 默认用 32 位小数
    for k in range(6):
        vec[k * 5 + OPS.index(arch[k])] = 1
    return vec


# ----------------------------------------------------------------------
# 预测器：一个三层的全连接神经网络（MLP）
#   30 个输入 -> 64 个神经元 -> 64 个神经元 -> 1 个输出（预测的分数）
# ReLU 的作用是把负数变成 0，让网络能学到"非直线"的关系。
# ----------------------------------------------------------------------
def make_mlp():
    return nn.Sequential(
        nn.Linear(30, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ----------------------------------------------------------------------
# 两种损失函数。pred 和 target 都是长度为 200 的一维张量（tensor）。
# ----------------------------------------------------------------------
def mse_loss(pred, target):
    """survey 公式 (4)：每个结构的 (真实值 - 预测值)^2，再取平均。"""
    return ((target - pred) ** 2).mean()


def pairwise_ranking_loss(pred, target):
    """survey 公式 (6)：对每一对"i 真实比 j 好"的结构，
    希望预测分数 pred[i] 至少比 pred[j] 高出 MARGIN，否则就罚。

    用矩阵一次算完所有 200 x 200 对，比写两层 for 循环快得多:
        diff[i, j]   = pred[i] - pred[j]
        better[i, j] = True 表示结构 i 的真实精度高于结构 j
    """
    diff = pred[:, None] - pred[None, :]
    better = target[:, None] > target[None, :]
    # relu(x) = max(0, x)。差距已经够大时罚 0；排反了或差距不够时罚得越来越多
    penalty = torch.relu(MARGIN - diff)
    return penalty[better].mean()


# ----------------------------------------------------------------------
# 训练一个 MLP 预测器。这几行是所有 PyTorch 训练代码的固定套路。
# ----------------------------------------------------------------------
def train_mlp(X_train, y_train, loss_fn, seed):
    torch.manual_seed(seed)  # 固定神经网络初始参数，保证可复现
    model = make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 把精度标准化成"平均 0、标准差 1"，神经网络对这种尺度的数学得更顺
    y_norm = (y_train - y_train.mean()) / y_train.std()
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_norm.astype(np.float32))

    for epoch in range(EPOCHS):
        pred = model(X_t).squeeze(1)  # 前向：算出 200 个预测值。squeeze 把 (200, 1) 变成 (200,)
        loss = loss_fn(pred, y_t)  # 算损失：预测得有多差
        optimizer.zero_grad()  # 清空上一轮留下的梯度
        loss.backward()  # 反向传播：算出每个参数该往哪个方向改
        optimizer.step()  # 按梯度更新参数
    return model


def predict_mlp(model, X):
    with torch.no_grad():  # 只预测不训练，告诉 PyTorch 不用记录梯度
        return model(torch.from_numpy(X)).squeeze(1).numpy()


# ----------------------------------------------------------------------
# 主实验：3 种方法 x 5 个种子
# ----------------------------------------------------------------------
archs, acc = load_dataset()
X = np.array([encode(a) for a in archs])
y = acc.astype(np.float32)

methods = ["Random forest", "MLP + MSE loss", "MLP + ranking loss"]
taus = {m: [] for m in methods}

for seed in SEEDS:
    perm = np.random.default_rng(seed).permutation(len(archs))
    train_idx, test_idx = perm[:N_TRAIN], perm[N_TRAIN:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X_train, y_train)
    preds = {
        "Random forest": rf.predict(X_test),
        "MLP + MSE loss": predict_mlp(train_mlp(X_train, y_train, mse_loss, seed), X_test),
        "MLP + ranking loss": predict_mlp(train_mlp(X_train, y_train, pairwise_ranking_loss, seed), X_test),
    }

    line = f"种子 {seed}:"
    for m in methods:
        tau, _ = kendalltau(preds[m], y_test)
        taus[m].append(tau)
        line += f"  {m} = {tau:.4f}"
    print(line)

print()
print(f"训练结构 {N_TRAIN} 个，5 次运行的 Kendall's Tau（平均值 ± 标准差）:")
for m in methods:
    print(f"  {m:20s} {np.mean(taus[m]):.4f} ± {np.std(taus[m]):.4f}")

# 柱状图：柱子高度是平均值，黑线是标准差
plt.figure(figsize=(6, 4))
plt.bar(methods, [np.mean(taus[m]) for m in methods],
        yerr=[np.std(taus[m]) for m in methods], capsize=6)
plt.ylabel("Kendall's Tau")
plt.title(f"NAS-Bench-201 CIFAR-10, {N_TRAIN} training architectures, {len(SEEDS)} seeds")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step3_compare.png", dpi=150)
print("柱状图已保存到 outputs/step3_compare.png")
