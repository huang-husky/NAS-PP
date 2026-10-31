"""把 NAS-Bench-101 的结构写成一小段 PyTorch 风格的代码，作为 LLM 的输入（和 arch_text.py 对 NB201 做的事一样）。

NB101 的一个 cell 有 2～7 个节点：x0 是输入，最后一个节点是输出，中间每个节点:
    先把所有输入加起来，再做一个操作（conv3x3 / conv1x1 / maxpool3x3）
输出节点把中间节点的结果拼接起来（torch.cat）；如果输入 x0 直接连到输出，再加上 x0。
例子（7 个节点）:
    class Cell(nn.Module):
        def forward(self, x0):
            x1 = self.conv3x3(x0)
            x2 = self.maxpool3x3(x1)
            x3 = self.conv3x3(x2)
            x4 = self.conv3x3(x0)
            x5 = self.conv1x1(x0 + x3 + x4)
            return torch.cat([x2, x5], dim=1)
真实网络里 x0 进中间节点前还有一个 1x1 投影（让通道数对齐），这里和 NB201 一样省略这类细节。

和 NB201 不同，这里每个结构的行数不一样（中间节点 0～5 个），所以只提取平均特征，不提分行特征。
"""

import csv
import json
from pathlib import Path

import numpy as np

NB101_CSV = Path(__file__).parent.parent / "predictor_lab" / "data" / "nasbench101.csv"
OPS = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]
OP_CODE = {"conv3x3-bn-relu": "self.conv3x3", "conv1x1-bn-relu": "self.conv1x1", "maxpool3x3": "self.maxpool3x3"}


def load_nb101():
    """返回 (结构列表, 数据表)。每个结构是 NNI 格式的字典：input1..inputN 和 op1..opN。"""
    with open(NB101_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [json.loads(r["arch"]) for r in rows], rows


def nb101_to_code(arch):
    n = len([k for k in arch if k.startswith("input")]) + 1  # 节点总数（含输入和输出）
    lines = ["class Cell(nn.Module):", "    def forward(self, x0):"]
    for k in range(1, n - 1):  # 中间节点
        summed = " + ".join(f"x{i}" for i in sorted(arch[f"input{k}"]))
        lines.append(f"        x{k} = {OP_CODE[arch[f'op{k}']]}({summed})")
    srcs = sorted(arch[f"input{n - 1}"])
    inner = [f"x{i}" for i in srcs if i != 0]
    if not inner:  # 输出只连着输入（只有 2 个节点的那个结构）
        out = "x0"
    else:
        out = inner[0] if len(inner) == 1 else f"torch.cat([{', '.join(inner)}], dim=1)"
        if 0 in srcs:
            out += " + x0"
    lines.append(f"        return {out}")
    return "\n".join(lines)


def encode_nb101(arch):
    """手工编码（以后评估用）：邻接矩阵上三角（7x7 的 21 位）+ 5 个节点各 3 种操作的 one-hot（15 位），
    以及 3 种操作的计数。节点少于 7 个时，空位全补 0。"""
    n = len([k for k in arch if k.startswith("input")]) + 1
    adj = np.zeros((7, 7), dtype=np.float32)
    ops = np.zeros((5, 3), dtype=np.float32)
    for k in range(1, n):
        dst = k if k < n - 1 else 6  # 输出节点统一放到第 6 号位置
        for i in arch[f"input{k}"]:
            adj[i, dst] = 1
    for k in range(1, n - 1):
        ops[k - 1, OPS.index(arch[f"op{k}"])] = 1
    onehot = np.concatenate([adj[np.triu_indices(7, 1)], ops.ravel()])
    return onehot, ops.sum(axis=0)


if __name__ == "__main__":
    archs, rows = load_nb101()
    for i in [0, 1]:
        print(nb101_to_code(archs[i]), f"\n# 测试精度 {rows[i]['test_acc']}\n")
    for a in archs:  # 找出节点最少的几种，确认边界情况也能写
        if len(a) <= 3:
            print(nb101_to_code(a), "\n")
