"""第 2 步：亲手做一个性能预测器（骨架，需要你补全 4 个 TODO）。

对应 survey 第 III-B-1 节的三个步骤:
    Step 1 采样并训练一批结构      -> 数据集已经替我们训练好了，只需要"抽样"（TODO 2）
    Step 2 把结构编码成向量        -> TODO 1
    Step 3 训练回归模型预测精度    -> TODO 3
最后用 Kendall's Tau 评价预测器排序排得准不准（TODO 4）。

运行方式（在 predictor_lab 文件夹下）:
    python step2_predictor.py
每补完一个 TODO 就运行一次，看到"[TODO x 通过]"再做下一个。
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

import checks
from nb201 import OPS, OUTPUT_DIR, load_dataset

archs, acc = load_dataset()


# ======================================================================
# TODO 1：编码（one-hot）
# ----------------------------------------------------------------------
# 回归模型只认数字，不认 'nor_conv_3x3' 这样的字符串，所以要先把结构变成向量。
# 最简单的办法：每条边用 5 个位置表示，选了哪种操作，就把对应的位置设成 1，其余为 0。
# 6 条边 x 5 个位置 = 长度 30 的向量。
#
# 例子：第 0 条边是 'skip_connect'，它在 OPS 里排第 1 个（从 0 数起），
#       所以 vec[0*5 + 1] = 1；
#       第 2 条边是 'nor_conv_3x3'，在 OPS 里排第 3 个，所以 vec[2*5 + 3] = 1。
#
# 会用到：
#     for k in range(6):       让 k 依次取 0, 1, 2, 3, 4, 5
#     arch[k]                  第 k 条边的操作名
#     OPS.index(名字)          这个操作名在 OPS 列表里排第几
# ======================================================================
def encode(arch):
    vec = np.zeros(30)
    for k in range(6):  # 依次处理 6 条边
        j = OPS.index(arch[k])  # 第 k 条边的操作在 OPS 里排第 j 个
        vec[k * 5 + j] = 1  # 第 k 条边占第 5k ~ 5k+4 这 5 个位置，把其中第 j 个设成 1
    return vec


checks.check_encode(encode)

# 把全部 15625 个结构都编码，得到形状为 (15625, 30) 的矩阵 X；标签 y 就是精度
X = np.array([encode(arch) for arch in archs])
y = acc
print("X 的形状:", X.shape, " y 的形状:", y.shape)


# ======================================================================
# TODO 2：划分训练集和测试集
# ----------------------------------------------------------------------
# 现实中训练一个结构要几个小时，所以预测器只能用很少的"已训练结构"来学习。
# 这里假装我们只训练得起 N_TRAIN 个结构，其余的全部用来检验预测得准不准。
#
# perm 是 0 ~ 15624 被随机打乱后的顺序。请取它的前 N_TRAIN 个作为训练集的下标，
# 剩下的作为测试集的下标。
#
# 会用到切片：
#     a[:3]    a 的前 3 个元素
#     a[3:]    从第 3 个开始到最后的所有元素
# ======================================================================
N_TRAIN = 200
rng = np.random.default_rng(seed=0)  # 固定随机种子，保证每次运行结果一样
perm = rng.permutation(len(archs))

train_idx = perm[:N_TRAIN]  # 打乱后的前 200 个下标：假装我们只训练得起这些结构
test_idx = perm[N_TRAIN:]  # 剩下的 15425 个下标：预测器从没见过，用来检验

checks.check_split(train_idx, test_idx, N_TRAIN, len(archs))

# numpy 数组可以直接用一组下标取出多行
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]


# ======================================================================
# TODO 3：训练预测器，并对测试集做预测
# ----------------------------------------------------------------------
# 回归模型先用随机森林（survey 第 9 页提到的非神经网络模型之一），它不用调参就能有不错的效果。
# scikit-learn 里所有模型的用法都一样，三步：
#     model = RandomForestRegressor(n_estimators=100, random_state=0)   创建模型
#     model.fit(训练集的 X, 训练集的 y)                                  训练
#     pred = model.predict(测试集的 X)                                  预测
# ======================================================================
model = RandomForestRegressor(n_estimators=100, random_state=0)  # 100 棵决策树组成的森林
model.fit(X_train, y_train)  # 训练：学习"编码 -> 精度"的映射
pred = model.predict(X_test)  # 预测：给每个没见过的结构估一个精度

checks.check_pred(model, pred, len(test_idx))


# ======================================================================
# TODO 4：用 Kendall's Tau 评价预测器
# ----------------------------------------------------------------------
# NAS 只需要知道"哪个结构更好"，所以评价预测器看的是排序是否一致，而不是数值差多少。
# Kendall's Tau：任取两个结构，预测的高低关系与真实的高低关系一致就加分，相反就扣分。
#     +1 表示排序完全一致，0 表示和乱猜一样，-1 表示完全相反。
#
# kendalltau(a, b) 会返回两个值：相关系数和 p 值。我们只要第一个：
#     tau, _ = kendalltau(a, b)
# ======================================================================
tau, _ = kendalltau(pred, y_test)  # 比较"预测的排序"和"真实的排序"，p 值用 _ 丢掉

checks.check_tau(tau, pred, y_test)


# ======================================================================
# 结果（已写好）
# ======================================================================
mse = np.mean((pred - y_test) ** 2)
print()
print(f"训练用了 {N_TRAIN} 个结构，在 {len(test_idx)} 个没见过的结构上测试")
print(f"Kendall's Tau = {tau:.4f}")
print(f"均方误差 MSE  = {mse:.4f}")

# 预测器认为最好的 10 个结构，真实排名是多少？
true_rank = np.argsort(np.argsort(-y_test)) + 1  # 每个测试结构的真实名次，1 表示最好
top10 = np.argsort(-pred)[:10]
print("预测器选出的前 10 名，它们的真实名次:", sorted(true_rank[top10].tolist()))

plt.scatter(y_test, pred, s=2, alpha=0.3)
plt.xlabel("true accuracy (%)")
plt.ylabel("predicted accuracy (%)")
plt.title(f"Random forest predictor, {N_TRAIN} training architectures, tau = {tau:.3f}")
plt.savefig(OUTPUT_DIR / "step2_scatter.png", dpi=150)
print("散点图已保存到 outputs/step2_scatter.png")

# ---- 做完之后的小实验（改一个数再运行，观察 tau 怎么变）----
# 1. 把 N_TRAIN 改成 50、1000，tau 分别是多少？这说明了预测器的什么痛点？
# 2. 把 seed=0 改成别的数，tau 变化大吗？论文里为什么总要报告多次运行的平均值？
# 3. 看散点图：精度很低的那些结构预测得准吗？这对 NAS 重要吗？
