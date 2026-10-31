"""第 1 步：用冻结的 LLM 给每个结构算一个特征向量，存到 features/ 文件夹。

做法：把结构写成代码文本 -> 送进 LLM（不训练，只做前向）-> 取最后一层隐藏状态，
每次前向同时算两种特征:
    平均特征   对所有 token 取平均，得到一个向量（Pythia-410M 1024 维，Qwen2.5-0.5B 896 维）
    分行特征   对 x1、x2、x3 三行代码分别取平均再拼起来（3 倍维度），保留"哪条边在哪一行"的位置信息

运行方式（在 llm_demo 文件夹下）:
    python embed.py --speed-test                         先测速：每个模型只编码 50 个结构
    python embed.py                                      NB201 全部 15625 个结构，所有模型
    python embed.py --only pythia410m_step0              只跑某个模型（多个用逗号隔开）
    python embed.py --anonymous                          算子名换成 op_A、op_B 这样没含义的名字
    python embed.py --shuffle-ops                        把 6 条边上的操作随机换位置（每种操作的个数不变）
    python embed.py --space transnas                     TransNAS-Bench-101 micro 的 4096 个结构
    python embed.py --space nb101 --pool 5000            NAS-Bench-101 里随机抽 5000 个结构（只存平均特征）
    python embed.py --limit 64 --out-dir features/_smoke 冒烟测试：只编码前 64 个结构，存到别的文件夹

已经算好的文件会自动跳过；结构池一旦确定（pool.npy），之后所有模型都用同一批结构。
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from arch_text import arch_to_code, load_dataset
from nb101_text import load_nb101, nb101_to_code
from models import MODEL_DIR, MODELS, has_weights
from progress import Progress

hf_logging.set_verbosity_error()  # 关掉加载模型时的长篇提示（比如"embed_out.weight 未使用"，不影响结果）
torch.set_flush_denormal(True)  # 把极小的数直接当 0 算，避免 CPU 在这类数上变慢

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "features"
TRANSNAS_CSV = HERE.parent / "predictor_lab" / "data" / "transnas_micro.csv"
POOL_SEED = 2026  # 抽结构池用的随机种子，固定下来保证可复现
CELL_LINES = [2, 3, 4]  # arch_to_code 生成的代码里，第 2、3、4 行（从 0 数）分别是 x1、x2、x3


def load_model(model_dir):
    """从本地文件夹读取分词器和模型（只要主干，不要最后预测下一个词的那一层）。"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:  # Pythia 没有填充符，借用"结束符"来补齐长度
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32)  # CPU 上用 32 位最快
    model.eval()  # 推理模式：关掉 dropout 之类只在训练时用的东西
    return tokenizer, model


def line_spans(text):
    """x1、x2、x3 三行在文本里的起止字符位置。"""
    starts, pos = [], 0
    for line in text.split("\n"):
        starts.append((pos, pos + len(line)))
        pos += len(line) + 1  # +1 是换行符
    return [starts[i] for i in CELL_LINES]


@torch.no_grad()  # 只做前向，不算梯度，省内存也更快
def embed_texts(tokenizer, model, texts, batch_size=16, name=None, per_line=True):
    """返回 (平均特征, 分行特征)。给了 name 就每 2 分钟打一行进度。per_line=False 时不算分行特征（返回 None）。"""
    means, per_line_out = [], []
    bar = Progress(len(texts), name) if name else None
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = batch.pop("offset_mapping")  # 每个 token 对应原文的哪几个字符；模型本身不需要它
        hidden = model(**batch).last_hidden_state  # (批大小, token 数, 隐藏维度)
        mask = batch["attention_mask"].unsqueeze(-1).float()  # 补齐出来的 token 不参与平均
        means.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1)).numpy())

        if not per_line:
            if bar:
                bar.update(len(chunk))
            continue
        # 分行特征：token 的起始字符落在哪一行，就算进哪一行的平均
        tok_start = offsets[:, :, 0]
        rows = []
        for b, text in enumerate(chunk):
            parts = []
            for lo, hi in line_spans(text):
                in_line = (tok_start[b] >= lo) & (tok_start[b] < hi) & batch["attention_mask"][b].bool()
                parts.append(hidden[b, in_line].mean(dim=0))
            rows.append(torch.cat(parts).numpy())
        per_line_out.append(np.stack(rows))
        if bar:
            bar.update(len(chunk))
    lines = np.concatenate(per_line_out).astype(np.float32) if per_line else None
    return np.concatenate(means).astype(np.float32), lines


def load_archs(space):
    """返回结构列表，每个结构是 6 个操作名（边的顺序和 NB201 一致）。"""
    if space == "nb201":
        return load_dataset()[0]
    if space == "nb101":
        return load_nb101()[0]
    with open(TRANSNAS_CSV, newline="", encoding="utf-8") as f:
        cols = ["op_0to1", "op_0to2", "op_1to2", "op_0to3", "op_1to3", "op_2to3"]
        return [[row[c] for c in cols] for row in csv.DictReader(f)]


def get_pool(feature_dir, n_total, pool_size):
    """抽出固定的一批结构下标。第一次运行时生成并保存，之后一直沿用。"""
    pool_file = feature_dir / "pool.npy"
    if pool_file.exists():
        pool = np.load(pool_file)
        if len(pool) != pool_size:
            raise SystemExit(f"已有的结构池是 {len(pool)} 个，和 --pool {pool_size} 不一致。"
                             f"要换大小请先把 {feature_dir} 文件夹改名备份。")
        return pool
    if pool_size >= n_total:
        pool = np.arange(n_total)
    else:
        pool = np.sort(np.random.default_rng(POOL_SEED).choice(n_total, size=pool_size, replace=False))
    np.save(pool_file, pool)
    return pool


def shuffle_ops(arch, index):
    """把 6 条边上的操作随机换位置：每种操作用了几次不变，但放在哪条边上被打乱。
    用结构编号当随机种子，所以每次运行打乱的结果都一样。"""
    perm = np.random.default_rng(index).permutation(len(arch))
    return [arch[p] for p in perm]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", choices=["nb201", "transnas", "nb101"], default="nb201")
    parser.add_argument("--pool", type=int, default=None, help="结构池大小，默认用全部结构")
    parser.add_argument("--only", type=str, default=None, help="只跑这些模型，多个用逗号隔开")
    parser.add_argument("--batch", type=int, default=32, help="每批编码的结构数；实测 32 在这台 CPU 上最快")
    parser.add_argument("--speed-test", action="store_true", help="每个模型只编码 50 个结构并计时")
    parser.add_argument("--anonymous", action="store_true", help="用匿名算子名（2×2 实验用）")
    parser.add_argument("--shuffle-ops", action="store_true", help="打乱操作所在的边（2×2 实验用）")
    parser.add_argument("--limit", type=int, default=None, help="只编码结构池里的前几个（冒烟测试用）")
    parser.add_argument("--out-dir", type=str, default=None, help="特征存到这里（冒烟测试时别覆盖正式结果）")
    args = parser.parse_args()

    archs = load_archs(args.space)
    names = args.only.split(",") if args.only else list(MODELS)
    feature_dir = Path(args.out_dir) if args.out_dir else FEATURE_DIR / ("" if args.space == "nb201" else args.space)
    feature_dir.mkdir(parents=True, exist_ok=True)

    nb101 = args.space == "nb101"
    if nb101 and (args.anonymous or args.shuffle_ops):
        raise SystemExit("NB101 暂不支持 --anonymous / --shuffle-ops")
    to_code = nb101_to_code if nb101 else arch_to_code

    if args.speed_test:
        texts = [to_code(a) for a in archs[:50]]
        print("每个模型编码 50 个结构的耗时：")
        for name in names:
            tok, model = load_model(MODEL_DIR / name)
            embed_texts(tok, model, texts[:4], args.batch, per_line=not nb101)  # 先热身一次，第一次运行总是偏慢
            t0 = time.time()
            embed_texts(tok, model, texts, args.batch, per_line=not nb101)
            per = (time.time() - t0) / len(texts)
            print(f"  {name:24s} 每个结构 {per:.3f} 秒  ->  全部 {len(archs)} 个约 {per * len(archs) / 60:.0f} 分钟",
                  flush=True)
        return

    pool = get_pool(feature_dir, len(archs), args.pool or len(archs))
    if args.limit:
        pool = pool[:args.limit]
    if nb101:
        texts = [nb101_to_code(archs[i]) for i in pool]
    else:
        cell = [shuffle_ops(archs[i], i) if args.shuffle_ops else archs[i] for i in pool]
        texts = [arch_to_code(a, anonymous=args.anonymous) for a in cell]
    suffix = ("_anon" if args.anonymous else "") + ("_shuf" if args.shuffle_ops else "")
    print(f"[{args.space}{suffix}] 结构池 {len(pool)} 个，第一个结构的输入文本：\n{texts[0]}\n", flush=True)

    for name in names:
        mean_file = feature_dir / f"{name}{suffix}.npy"
        line_file = feature_dir / f"{name}{suffix}_lines.npy"
        if mean_file.exists() and (nb101 or line_file.exists()):
            print(f"[跳过] {mean_file.name} 已存在")
            continue
        if not (MODEL_DIR / name / ".download_done").exists() or not has_weights(MODEL_DIR / name):
            print(f"[跳过] {name} 还没下载完，下载好后重新运行本脚本即可补上")
            continue
        t0 = time.time()
        tok, model = load_model(MODEL_DIR / name)
        means, lines = embed_texts(tok, model, texts, args.batch, name=f"{args.space} {name}{suffix}",
                                   per_line=not nb101)
        if not mean_file.exists():  # 已有的平均特征不覆盖（之前的评估结果用的就是它）
            np.save(mean_file, means)
        if lines is not None:
            np.save(line_file, lines)
        print(f"[完成] {name}{suffix}  平均特征 {means.shape}，分行特征 {None if lines is None else lines.shape}  "
              f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        del model


if __name__ == "__main__":
    main()
