"""词频特征（bag-of-words）：直接检验"平均池化的 LLM 特征 ≈ 代码里各个词出现了几次"。

做法：用 Pythia 自己的分词器，把每个结构的代码文本切成 token（和送进模型时一模一样），
数每种 token 出现了几次，得到一个计数向量。所有结构都一样的 token（比如 "class"）没有信息，去掉。
这里不挑词、不看精度，模型看到什么就数什么，所以是一个事先定好的检验。

如果解释是对的，词频特征的学习曲线应该和随机初始化 Pythia 的平均特征差不多：
小样本时一样好（甚至更好），样本多时卡在差不多的位置。

存到和 LLM 特征相同的文件夹，名字叫 bow.npy，评估脚本 step2.py 会把它当成一种特征来源。

运行方式（在 llm_demo 文件夹下）:
    python make_bow.py
"""

from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from arch_text import arch_to_code
from embed import FEATURE_DIR, load_archs
from models import MODEL_DIR
from nb101_text import nb101_to_code

TOKENIZER = MODEL_DIR / "pythia410m_step0"  # 所有 Pythia 存档用的是同一个分词器


def bow(texts, tokenizer):
    counts = [Counter(tokenizer(t)["input_ids"]) for t in texts]
    vocab = sorted(set().union(*counts))
    X = np.array([[c.get(v, 0) for v in vocab] for c in counts], dtype=np.float32)
    keep = X.std(axis=0) > 0  # 去掉所有结构都一样的 token
    return X[:, keep], [tokenizer.decode([v]) for v, k in zip(vocab, keep) if k]


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for space, folder in [("nb201", FEATURE_DIR), ("transnas", FEATURE_DIR / "transnas"),
                          ("nb101", FEATURE_DIR / "nb101")]:
        archs = load_archs(space)
        pool = np.load(folder / "pool.npy")  # 和 LLM 特征用同一批结构、同一个顺序
        to_code = nb101_to_code if space == "nb101" else arch_to_code
        X, words = bow([to_code(archs[i]) for i in pool], tokenizer)
        np.save(folder / "bow.npy", X)
        print(f"[{space}] {X.shape[0]} 个结构，{X.shape[1]} 种会变的 token：{words}")


if __name__ == "__main__":
    main()
