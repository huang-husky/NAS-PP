"""把 NAS-Bench-201 的结构写成一小段 PyTorch 风格的代码，作为 LLM 的输入。

NAS-Bench-201 的一个 cell 有 4 个节点（x0 ~ x3），6 条边，每条边选一个操作:
    x1 = 边(0->1)(x0)
    x2 = 边(0->2)(x0) + 边(1->2)(x1)
    x3 = 边(0->3)(x0) + 边(1->3)(x1) + 边(2->3)(x2)
下面的函数就是按这个规则拼出代码文本。
"""

import sys
from pathlib import Path

import numpy as np

# 复用入门实验里写好的数据读取工具
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_lab"))
from nb201 import EDGES, OPS, load_dataset  # noqa: E402

# 操作名 -> 代码里的写法（有意义的名字）
OP_CODE = {
    "none": "self.zero",
    "skip_connect": "self.identity",
    "nor_conv_1x1": "self.conv1x1",
    "nor_conv_3x3": "self.conv3x3",
    "avg_pool_3x3": "self.avgpool3x3",
}

# 匿名版本：把操作名换成没有含义的符号，用于以后的 2×2 因子实验
OP_CODE_ANON = {op: f"self.op_{chr(ord('A') + i)}" for i, op in enumerate(OPS)}


def arch_to_code(arch, anonymous=False):
    """arch 是 6 个操作名组成的列表（顺序与 EDGES 一致），返回一段代码文本。"""
    names = OP_CODE_ANON if anonymous else OP_CODE
    # 按终点节点把边分组：哪些边流入 x1、x2、x3
    terms = {1: [], 2: [], 3: []}
    for (src, dst), op in zip(EDGES, arch):
        terms[dst].append(f"{names[op]}(x{src})")
    lines = [
        "class Cell(nn.Module):",
        "    def forward(self, x0):",
        f"        x1 = {' + '.join(terms[1])}",
        f"        x2 = {' + '.join(terms[2])}",
        f"        x3 = {' + '.join(terms[3])}",
        "        return x3",
    ]
    return "\n".join(lines)


def one_hot(arch):
    """入门实验里的手工编码：6 条边 x 5 种操作 = 30 维 0/1 向量。"""
    vec = np.zeros(30, dtype=np.float32)
    for k in range(6):
        vec[k * 5 + OPS.index(arch[k])] = 1
    return vec


if __name__ == "__main__":
    archs, acc = load_dataset()
    print(arch_to_code(archs[1]))
    print()
    print(arch_to_code(archs[1], anonymous=True))
    print(f"\n这个结构在 CIFAR-10 上的精度: {acc[1]:.2f}")
