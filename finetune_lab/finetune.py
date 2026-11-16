"""微调小语言模型做性能预测（因子阶梯的 A0 / A1，后续因子在这个脚本上逐个加）。

A0：Pythia-70M（预训练初始化）+ 最后一层隐藏状态对所有 token 取平均 + 一个线性层输出一个数
    + 标准化后的均方误差 + code 写法的输入。全部参数一起训练（全量微调）。
A1：同上，但换成"第 0 步存档"，也就是同样的结构、随机初始化。

协议见 PROTOCOL.md：按功能等价类切分；开发种子 100～102 用来选超参数，评估种子 0～4 用来下结论。

用法（在 finetune_lab 文件夹下）
    python -I finetune.py --quick                              冒烟测试（1 分钟左右）
    python -I finetune.py --model pythia70m_step143000 --tag A0 --n 200 --seeds 100,101,102 --lr 2e-4 --epochs 30
结果追加写入 results/ft_<tag>.csv；同一个 (tag, 种子, n) 已经有结果就跳过，所以中断后可以接着跑。
"""

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filte22rwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_demo"))
from protocol import Data, evaluate  # noqa: E402
from arch_text import arch_to_code  # noqa: E402
from formats import cell_text  # noqa: E402
from nb201 import OPS  # noqa: E402

MODEL_DIR = HERE.parent / "llm_demo" / "models"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
N_DIGITS = 3


# ======================================================================================
# 第 1 段：模型 = 语言模型主体 + 一个线性回归头
# ======================================================================================
class Regressor(nn.Module):
    def __init__(self, path, head="mse"):
        super().__init__()
        self.head_type = head
        from transformers import AutoModel
        self.lm = AutoModel.from_pretrained(path, dtype=torch.float32)
        # Pythia 的隐藏状态里有几个维度数值特别大，直接接线性层第一步输出就会爆炸（冒烟测试里损失到过 201）。
        # 所以先做一次不带可学习参数的层归一化，并把回归头初始化为 0（一开始预测均值，再慢慢学）。
        self.norm = nn.LayerNorm(self.lm.config.hidden_size, elementwise_affine=False)
        self.head = nn.Linear(self.lm.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # digit 头：3 位数字，依次预测，后一位以前面数字的 one-hot 为条件
        H = self.lm.config.hidden_size
        self.dig = nn.ModuleList([nn.Linear(H + 10 * k, 10) for k in range(N_DIGITS)])
        for m in self.dig:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids, mask):
        h = self.lm(input_ids=ids, attention_mask=mask).last_hidden_state        # (批, token 数, 维度)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)                                      # 只对真实 token 取平均
        return self.norm(pooled)                                                # 特征，各头自己再接

    def scalar(self, feat):                                                     # mse / rank 头
        return self.head(feat).squeeze(-1)

    def digit_loss(self, feat, digits):                                         # digits: (批, 3) 真值，teacher forcing
        loss, prefix = 0.0, []
        for k in range(N_DIGITS):
            logits = self.dig[k](torch.cat([feat] + prefix, dim=1))
            loss = loss + nn.functional.cross_entropy(logits, digits[:, k])
            prefix.append(nn.functional.one_hot(digits[:, k], 10).float())
        return loss

    def digit_score(self, feat):                                                # 贪心解码；打分 = 各位期望值 × 位权
        score, prefix = 0.0, []
        for k in range(N_DIGITS):
            p = torch.softmax(self.dig[k](torch.cat([feat] + prefix, dim=1)), dim=1)
            score = score + (p * torch.arange(10.0)).sum(1) * 10.0 ** (N_DIGITS - 1 - k)
            prefix.append(nn.functional.one_hot(p.argmax(1), 10).float())
        return score


def encode(tok, texts):
    """分词；返回每条文本的 token 编号列表。"""
    return [tok(t)["input_ids"] for t in texts]


def batchify(seqs, idx, pad_id):
    """把若干条序列补齐成一个批；右侧补齐，用注意力掩码标出真实 token。"""
    L = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
        mask[r, :len(seqs[i])] = 1
    return ids, mask


# ======================================================================================
# 第 2 段：训练与预测
# ======================================================================================
def train_and_eval(model_name, seqs, pad_id, d, tr, te, args, seed):
    # 训练预算按"优化步数"给（CPU 上每步约 1.5 秒）：样本少的设置多跑几轮，样本多的少跑几轮，每次大约 args.steps 步
    args.epochs_eff = args.epochs or math.ceil(args.steps / math.ceil(len(tr) / args.bs))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Regressor(MODEL_DIR / model_name, args.head)
    if args.init:                                  # 读入预训练检查点（语言模型主体；回归头仍是新的、零初始化）
        model.lm.load_state_dict(torch.load(HERE / "models" / args.init))
    if args.fmt == "merged":                       # 5 个新"操作词"占用词表末尾空位，嵌入初始化为已有嵌入的平均
        emb = model.lm.get_input_embeddings().weight
        assert emb.shape[0] >= args.V + 5, "嵌入表没有空位"
        with torch.no_grad():
            emb[args.V:args.V + 5] = emb[:args.V].mean(0)
    y = d.y[tr]
    mu, sd = y.mean(), y.std() + 1e-9
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    assert 1 < d.y.max() <= 100, "精度应以百分数给出"
    v = np.clip(np.round(y * 10), 0, 10 ** N_DIGITS - 1).astype(int)           # 精度×10 取整，3 位数字
    dig = torch.tensor(np.stack([(v // 10 ** (N_DIGITS - 1 - k)) % 10 for k in range(N_DIGITS)], 1), dtype=torch.long)
    yraw = torch.tensor(y, dtype=torch.float32)

    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs_eff
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    rng = np.random.default_rng(seed)
    model.train()
    step, t0, last_loss = 0, time.time(), float("nan")
    for ep in range(args.epochs_eff):
        perm = rng.permutation(len(tr))
        for b in range(steps_per_epoch):
            sel = perm[b * args.bs:(b + 1) * args.bs]
            ids, mask = batchify(seqs, tr[sel], pad_id)
            feat = model(ids, mask)
            if args.head == "mse":
                loss = ((model.scalar(feat) - yt[sel]) ** 2).mean()
            elif args.head == "digit":
                loss = model.digit_loss(feat, dig[sel])
            else:                                                                # rank：批内成对排序损失
                sc, yy = model.scalar(feat), yraw[sel]
                diff = sc[:, None] - sc[None, :]
                pair = yy[:, None] > yy[None, :]
                loss = nn.functional.softplus(-diff[pair]).mean() if pair.any() else sc.sum() * 0
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            last_loss = loss.item()
        if (ep + 1) % max(1, args.epochs_eff // 5) == 0:
            print(f"      第 {ep + 1}/{args.epochs_eff} 轮，训练损失 {last_loss:.3f}，已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    train_min = (time.time() - t0) / 60

    model.eval()
    order = np.argsort([len(seqs[i]) for i in te])               # 按长度排序，减少补齐浪费
    pred = np.zeros(len(te))
    with torch.no_grad():
        for b in range(0, len(te), 64):
            sel = order[b:b + 64]
            ids, mask = batchify(seqs, te[sel], pad_id)
            feat = model(ids, mask)
            pred[sel] = (model.digit_score(feat) if args.head == "digit" else model.scalar(feat)).numpy()
    if args.head == "mse":
        pred = pred * sd + mu                                      # 只影响尺度，不影响排序
    return evaluate(pred, d.y[te]), train_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia70m_step143000", help="llm_demo/models/ 下的文件夹名")
    ap.add_argument("--tag", default="A0")
    ap.add_argument("--n", type=int, nargs="+", default=[200])
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=300, help="每次训练大约多少个优化步（默认 300）；轮数由它和 n 推出")
    ap.add_argument("--epochs", type=int, default=None, help="直接指定轮数（会覆盖 --steps）")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--init", default=None, help="finetune_lab/models/ 下的检查点文件名（因子 D 的回归预训练）")
    ap.add_argument("--fmt", default="code", choices=["code", "posword", "merged"], help="输入写法（因子 C）")
    ap.add_argument("--head", default="mse", choices=["mse", "digit", "rank"], help="输出头（因子 B）")
    ap.add_argument("--quick", action="store_true", help="冒烟测试：n=50，1 个种子，3 轮，测试集只取 300 个")
    args = ap.parse_args()
    if args.quick:
        args.n, args.seeds, args.epochs, args.tag = [50], "100", 3, "_quick"
    seeds = [int(s) for s in args.seeds.split(",")]
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d = Data()
    args.V = len(tok)
    if args.fmt == "merged":        # 每个操作 = 1 个新词，一个结构 = 6 个词（位置由顺序表示）
        seqs = [[args.V + int(k) for k in row] for row in d.ops]
    else:
        texts = [cell_text([OPS[k] for k in row], args.fmt) for row in d.ops]
        seqs = encode(tok, texts)
    print(f"模型 {args.model}；{args.fmt} 写法每个结构 {np.mean([len(s) for s in seqs]):.0f} 个 token"
          f"（最长 {max(len(s) for s in seqs)}）", flush=True)

    out_file = OUT / f"ft_{args.tag}.csv"
    done = set()
    if out_file.exists():
        old = pd.read_csv(out_file)
        done = set(zip(old.seed, old.n, old.lr, old.budget))
    budget = f"epochs{args.epochs}" if args.epochs else f"steps{args.steps}"
    for seed in seeds:
        for n in args.n:
            if (seed, n, args.lr, budget) in done:
                print(f"跳过已完成：种子 {seed}, n={n}", flush=True)
                continue
            tr, te = d.split(seed, n)
            if args.quick:
                te = te[:300]
            print(f"[{args.tag}] 种子 {seed}，n={n}，lr={args.lr}，预算 {budget}", flush=True)
            m, minutes = train_and_eval(args.model, seqs, pad_id, d, tr, te, args, seed)
            row = dict(tag=args.tag, model=args.model, head=args.head, fmt=args.fmt, init=args.init, seed=seed, n=n, lr=args.lr, budget=budget,
                       epochs=args.epochs_eff, bs=args.bs, minutes=round(minutes, 2), **m)
            pd.DataFrame([row]).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
            print(f"   -> tau {m['tau']:.3f}  rho {m['rho']:.3f}  regret@10 {m['regret10']:.2f}  用时 {minutes:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
