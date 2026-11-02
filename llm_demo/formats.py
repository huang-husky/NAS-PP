"""规律 2 用的几种"写法"：同一个结构，换不同的文本写法送进 LLM。

五种写法（三个搜索空间都有）:
    code      PyTorch 风格代码（第一阶段一直用的）
    edges     边列表：每条边一行，写明从哪个节点到哪个节点、用什么操作
    sorted    只列操作：把所有操作按字母顺序列出来，位置信息完全删掉
    nl        自然语言：用英文句子描述每个节点怎么算
    posword   位置词：每个"位置 + 操作"组合用一个专门的英文单词（比如第 0 条边的 conv3x3 写成 tiger），
              这样一个词就同时说明了"在哪"和"是什么"，词频就等于完整的 one-hot

规律 2 的预言：每种写法下，LLM 特征的表现 ≈ 这种写法的词频特征的表现。
前四种写法的词频里只有"操作计数"（位置要么被删掉，要么被拆成单独的数字），应该都有上限；
posword 的词频里有完整的位置信息，应该没有上限。

运行方式（在 llm_demo 文件夹下）:
    python formats.py        打印每个空间、每种写法的例子
"""

from arch_text import arch_to_code
from nb101_text import OPS as NB101_OPS
from nb101_text import nb101_to_code

FORMATS = ["code", "edges", "sorted", "nl", "posword"]

CELL_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]  # NB201 和 TransNAS（少一个池化）
CELL_EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
CELL_WORD = {"none": "zero", "skip_connect": "identity", "nor_conv_1x1": "conv1x1",
             "nor_conv_3x3": "conv3x3", "avg_pool_3x3": "avgpool3x3"}
NB101_WORD = {"conv3x3-bn-relu": "conv3x3", "conv1x1-bn-relu": "conv1x1", "maxpool3x3": "maxpool3x3"}

# posword 用的词：都是常见英文名词，而且在 Pythia 的分词器里各自只占一个 token（make 时检查过），
# 和操作、位置没有任何含义上的联系
WORDS = ["apple", "river", "tiger", "chair", "cloud", "piano", "lemon", "horse", "candle", "garden",
         "pencil", "rocket", "bottle", "forest", "violin", "mirror", "basket", "desert", "jacket", "ladder",
         "magnet", "needle", "orange", "pepper", "rabbit", "saddle", "tomato", "wallet", "anchor", "barrel",
         "dragon", "eagle", "guitar", "hammer", "island", "jungle", "marble", "nickel", "quartz", "salmon",
         "turtle", "umbrella", "valley", "yogurt", "zebra", "bridge", "castle", "donkey", "engine", "feather"]


def _join(items):
    """['a'] -> 'a'；['a','b'] -> 'a and b'；['a','b','c'] -> 'a, b and c'"""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- NB201 / TransNAS（6 条边的 cell）

def cell_text(arch, fmt):
    """arch 是 6 个操作名（边的顺序见 CELL_EDGES）。"""
    if fmt == "code":
        return arch_to_code(arch)
    if fmt == "edges":
        return "\n".join(f"edge {s}->{d}: {CELL_WORD[op]}" for (s, d), op in zip(CELL_EDGES, arch))
    if fmt == "sorted":
        return "operations: " + " ".join(sorted(CELL_WORD[op] for op in arch))
    if fmt == "nl":
        parts = ["The cell has four nodes."]
        for dst in (1, 2, 3):
            terms = [f"{CELL_WORD[op]} of node {s}" for (s, d), op in zip(CELL_EDGES, arch) if d == dst]
            if len(terms) == 1:
                parts.append(f"Node {dst} applies {terms[0].replace(' of ', ' to ')}.")
            else:
                parts.append(f"Node {dst} adds {_join(terms)}.")
        parts.append("The output is node 3.")
        return " ".join(parts)
    if fmt == "posword":
        return "cell: " + " ".join(WORDS[k * len(CELL_OPS) + CELL_OPS.index(op)] for k, op in enumerate(arch))
    if fmt == "cole":
        return cole_text(arch)
    raise ValueError(fmt)


# ---------------------------------------------------------------- COLE 论文的写法（只用于 NB201 / TransNAS）
# 照着 COLE（Somu 等，GECCO 2026）公开代码 stringify_utils.py 里的 to_pytorch_code 重写，
# 用它的 'helper' 模式（论文图 2 的样子）：每条边一个专门的名字 self.op_<起点>_<终点>，
# 无连接（none）直接省略，跳连写成 "+ node_<起点>"。
# 这样"哪条边放了什么"会改变 op_0_2、node_1 这些词出现的次数：位置信息进到了词频里。

COLE_HELPER = "\n".join([
    "class ReLU_Conv2d_BatchNorm(nn.Module):",
    "    def __init__(self, channels, kernel_size, stride, padding):",
    "        super().__init__()",
    "        self.op = nn.Sequential(",
    "            nn.ReLU(inplace=False),",
    "            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False),",
    "            nn.BatchNorm2d(channels)",
    "        )",
    "    ",
    "    def forward(self, x):",
    "        return self.op(x)"])
COLE_OP = {"avg_pool_3x3": "nn.AvgPool2d(kernel_size=3, stride=1, padding=1)",
           "nor_conv_1x1": "ReLU_Conv2d_BatchNorm(channels, kernel_size=1, stride=1, padding=0)",
           "nor_conv_3x3": "ReLU_Conv2d_BatchNorm(channels, kernel_size=3, stride=1, padding=1)"}


def cole_text(arch):
    edges = [(s, d, op) for (s, d), op in zip(CELL_EDGES, arch)]
    lines = ["class Cell(nn.Module):", "    def __init__(self, channels):", "        super().__init__()"]
    lines += [f"        self.op_{s}_{d} = {COLE_OP[op]}" for s, d, op in edges if op in COLE_OP]
    lines += ["", "    def forward(self, x):", "        node_0 = x"]
    for target in (1, 2, 3):
        incoming = [f"node_{s}" if op == "skip_connect" else f"self.op_{s}_{d}(node_{s})"
                    for s, d, op in edges if d == target and op != "none"]
        lines.append(f"        node_{target} = " + (" + ".join(incoming) if incoming else "torch.zeros_like(node_0)"))
    lines.append("        return node_3")
    return COLE_HELPER + "\n\n" + "\n".join(lines)


# ---------------------------------------------------------------- NB101（任意连线的 DAG）

def _nb101_parts(arch):
    n = len([k for k in arch if k.startswith("input")]) + 1
    ops = {k: NB101_WORD[arch[f"op{k}"]] for k in range(1, n - 1)}
    edges = [(i, k) for k in range(1, n) for i in sorted(arch[f"input{k}"])]
    return n, ops, edges


def nb101_text(arch, fmt):
    if fmt == "code":
        return nb101_to_code(arch)
    n, ops, edges = _nb101_parts(arch)
    if fmt == "edges":
        lines = [f"node {k}: {w}" for k, w in ops.items()]
        lines += [f"output: node {n - 1}", "edges: " + " ".join(f"{i}->{j}" for i, j in edges)]
        return "\n".join(lines)
    if fmt == "sorted":
        return "operations: " + (" ".join(sorted(ops.values())) or "none")
    if fmt == "nl":
        parts = [f"The cell has {n} nodes."]
        for k, w in ops.items():
            srcs = [f"{i}" for i in sorted(arch[f"input{k}"])]
            what = f"node {srcs[0]}" if len(srcs) == 1 else f"the sum of nodes {_join(srcs)}"
            parts.append(f"Node {k} applies {w} to {what}.")
        srcs = sorted(arch[f"input{n - 1}"])
        inner = [str(i) for i in srcs if i != 0]
        if not inner:
            parts.append("The output is node 0.")
        else:
            out = f"The output {'is node' if len(inner) == 1 else 'concatenates nodes'} {_join(inner)}"
            parts.append(out + (" and adds node 0." if 0 in srcs else "."))
        return " ".join(parts)
    if fmt == "posword":
        # 中间节点 k 的操作 -> 词 (k-1)*3+op；边 (i, j) -> 上三角编号（输出节点统一当作 6 号），和 encode_nb101 一一对应
        words = [WORDS[(k - 1) * 3 + NB101_OPS.index(arch[f"op{k}"])] for k in ops]
        tri = {(i, j): t for t, (i, j) in enumerate((i, j) for i in range(7) for j in range(i + 1, 7))}
        words += [WORDS[15 + tri[(i, 6 if j == n - 1 else j)]] for i, j in edges]
        return "cell: " + " ".join(words)
    raise ValueError(fmt)


def to_text(space, arch, fmt):
    return nb101_text(arch, fmt) if space == "nb101" else cell_text(arch, fmt)


if __name__ == "__main__":
    from embed import load_archs
    for space in ["nb201", "transnas", "nb101"]:
        arch = load_archs(space)[1234]
        print(f"================ {space}  结构 {arch}")
        for fmt in FORMATS:
            print(f"---- {fmt}\n{to_text(space, arch, fmt)}\n")
