import matplotlib.pyplot as plt
import numpy as np

from nb201 import EDGES, OPS, OUTPUT_DIR, load_dataset, to_arch_str

archs, acc = load_dataset()

# ---- 1. 数据长什么样 ----
print("结构总数:", len(archs))
print("精度数组的形状:", acc.shape)
print()
print("第 0 个结构的 6 个操作:", archs[0])
print("标准写法:", to_arch_str(archs[0]))
print("它的精度:", acc[0])
print()

# 每条边上是什么操作
for (src, dst), op in zip(EDGES, archs[0]):
    print(f"  节点 {src} -> 节点 {dst}: {op}")
print()

# ---- 2. 精度的整体分布 ----
print(f"最高精度 {acc.max():.2f}，最低精度 {acc.min():.2f}，平均 {acc.mean():.2f}")
best = int(np.argmax(acc))  # argmax 返回最大值所在的位置
worst = int(np.argmin(acc))
print("最好的结构:", to_arch_str(archs[best]))
print("最差的结构:", to_arch_str(archs[worst]))
print()

# ---- 3. 哪种操作更容易带来高精度 ----
# 对每种操作，统计"用了它的结构"的平均精度
for op in OPS:
    used = np.array([op in arch for arch in archs])  # 一个 True/False 数组
    print(f"  用到 {op:13s} 的结构有 {used.sum():5d} 个，平均精度 {acc[used].mean():.2f}")
print()

# ---- 4. 画精度直方图 ----
plt.hist(acc, bins=100)
plt.xlabel("CIFAR-10 accuracy (%)")
plt.ylabel("number of architectures")
plt.title("NAS-Bench-201: accuracy of all 15625 architectures")
plt.savefig(OUTPUT_DIR / "step1_hist.png", dpi=150)
print("直方图已保存到 outputs/step1_hist.png")

# ---- 思考题（不用写代码，想清楚后可以来问我）----
# 1. 直方图里大部分结构的精度集中在哪个区间？左边那一小撮精度极低的结构，6 条边有什么共同点？
#    （提示：打印 archs[worst] 看看，再想想 'none' 表示什么）
# 2. 如果精度都挤在 85~94 这么窄的范围里，预测器把每个结构都预测成 90，
#    均方误差会很小。这样的预测器对 NAS 有用吗？为什么 survey 强调"排序"？
