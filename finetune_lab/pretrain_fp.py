"""D 步（数据技巧）：用 FLOPs 和参数量做"回归预训练"。

做什么：让模型读结构的 code 写法，回归预测 log(1+FLOPs) 和 log(1+参数量)（各自标准化）。
这两个量是结构本身算出来的，不需要训练一个网络，也不涉及任何精度标签。
结果：把预训练后的语言模型主体权重存成检查点，之后 finetune.py --init 读入，再用少量精度标签微调。

注意（写进 PROTOCOL.md）：预训练用全部 15625 个结构的文字，其中包含测试结构。这不是标签泄漏（只用了 FLOPs 和参数量），
但属于"用了无标签的测试结构"；在实际 NAS 里搜索空间的结构都是已知的，这是合理的设定。

用法
    python -I pretrain_fp.py --quick                 冒烟测试（几十秒）
    python -I pretrain_fp.py --steps 1000            正式预训练（约 25 分钟），存 models/fp_pretrained.pt
"""

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from finetune import MODEL_DIR, Regressor, batchify, encode  # noqa: E402
from finetune import OPS, arch_to_code, Data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia70m_step143000")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="fp_pretrained.pt")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.out = 20, "fp_quick.pt"
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    torch.manual_seed(0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d = Data()
    seqs = encode(tok, [arch_to_code([OPS[k] for k in row]) for row in d.ops])
    tgt = np.stack([np.log1p(d.flops), np.log1p(d.params)], axis=1)
    tgt = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-9)                      # 各自标准化
    tgt = torch.tensor(tgt, dtype=torch.float32)

    model = Regressor(MODEL_DIR / args.model)
    model.fp_head = nn.Linear(model.lm.config.hidden_size, 2)            # 预训练专用的两输出头，不保存
    nn.init.zeros_(model.fp_head.weight)
    nn.init.zeros_(model.fp_head.bias)
    params = list(model.lm.parameters()) + list(model.fp_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(len(seqs)) for _ in range(math.ceil(args.steps * args.bs / len(seqs)))])
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        sel = order[step * args.bs:(step + 1) * args.bs]
        ids, mask = batchify(seqs, sel, pad_id)
        loss = ((model.fp_head(model(ids, mask)) - tgt[sel]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, args.steps // 20) == 0:
            el = time.time() - t0
            bar = "#" * int(20 * (step + 1) / args.steps) + "-" * (20 - int(20 * (step + 1) / args.steps))
            print(f"[{bar}] {step + 1}/{args.steps} 步，损失 {loss.item():.4f}，已用 {el / 60:.1f} 分钟，"
                  f"预计剩余 {el / (step + 1) * (args.steps - step - 1) / 60:.1f} 分钟", flush=True)

    # 在 1000 个随机结构上看预训练学得怎么样（用它们的 FLOPs，不是精度）
    model.eval()
    chk = np.random.default_rng(1).choice(len(seqs), 1000, replace=False)
    with torch.no_grad():
        pred = torch.cat([model.fp_head(model(*batchify(seqs, chk[b:b + 100], pad_id))) for b in range(0, 1000, 100)])
    r = [float(np.corrcoef(pred[:, k].numpy(), tgt[chk, k].numpy())[0, 1]) for k in range(2)]
    print(f"预训练结束：FLOPs 预测相关系数 {r[0]:.3f}，参数量 {r[1]:.3f}（在 1000 个随机结构上；这些结构训练时可能见过）", flush=True)
    (HERE / "models").mkdir(exist_ok=True)
    torch.save(model.lm.state_dict(), HERE / "models" / args.out)
    print("已保存", HERE / "models" / args.out, flush=True)


if __name__ == "__main__":
    main()
