"""规律 4：取特征的方式会不会改变"冻结特征 ≈ 词频"？

之前的特征都是"最后一层、所有 token 取平均"。这里换两个维度:
    哪一层     第 0 层（词嵌入，还没经过任何注意力）、第 6、12、18、24 层（410M 一共 24 层）
    怎么取     mean：所有 token 取平均；last：只取最后一个 token
预言:
    第 0 层 + mean 在数学上就是"词频 × 嵌入矩阵"，应该和词频一模一样
    mean 在每一层都应该贴近词频（平均操作会把位置信息抹掉）
    last 不是"一袋词"：最后一个 token 通过注意力按顺序看过前面所有 token，可能保留位置信息、突破天花板

只用 code 写法、两个 410M Pythia（随机 / 训练完成），三个空间的结构池和规律 2 相同。
一次前向就能拿到所有层，所以和只取最后一层一样快。
特征存到 features/layers/<空间>/<模型>_L<层>_<mean|last>.npy；结构池和词频从 features/formats/ 复制过来。

运行方式（在 llm_demo 文件夹下）:
    python run_layers.py --limit 32 --out features/_smoke_layers    冒烟测试
    python run_layers.py
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from embed import FEATURE_DIR, load_archs, load_model
from formats import to_text
from models import MODEL_DIR
from progress import Progress

SPACES = ["nb201", "transnas", "nb101"]
MODELS = ["pythia410m_step0", "pythia410m_step143000"]
LAYERS = [0, 6, 12, 18, 24]
FMT_DIR = FEATURE_DIR / "formats"


@torch.no_grad()
def embed_layers(tokenizer, model, texts, layers, batch_size, name):
    """返回 {(层, 'mean'|'last'): 特征矩阵}。"""
    out = {(l, p): [] for l in layers for p in ["mean", "last"]}
    bar = Progress(len(texts), name)
    tokenizer.padding_side = "right"  # 补齐放在右边，这样每段文本最后一个真实 token 的位置 = 长度 − 1
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True)
        hidden = model(**batch, output_hidden_states=True).hidden_states  # 第 0 个是词嵌入，之后每层一个
        mask = batch["attention_mask"].unsqueeze(-1).float()
        last = batch["attention_mask"].sum(dim=1) - 1
        for l in layers:
            h = hidden[l]
            out[l, "mean"].append(((h * mask).sum(dim=1) / mask.sum(dim=1)).numpy())
            out[l, "last"].append(h[torch.arange(len(chunk)), last].numpy())
        bar.update(len(chunk))
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(FEATURE_DIR / "layers"))
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    for name in args.models.split(","):
        tok, model = load_model(MODEL_DIR / name)
        for space in SPACES:
            out_dir = Path(args.out) / space
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in ["pool.npy", "code_bow.npy"]:  # 和规律 2 用同一批结构、同一份词频
                if not (out_dir / f).exists():
                    if args.limit:  # 冒烟测试：结构池和词频也只留前几个，评估脚本才能对得上
                        np.save(out_dir / f, np.load(FMT_DIR / space / f)[:args.limit])
                    else:
                        shutil.copy(FMT_DIR / space / f, out_dir / f)
            files = {(l, p): out_dir / f"{name}_L{l}_{p}.npy" for l in LAYERS for p in ["mean", "last"]}
            if all(f.exists() for f in files.values()):
                print(f"[跳过] {space} {name} 已全部存在", flush=True)
                continue
            archs = load_archs(space)
            pool = np.load(out_dir / "pool.npy")[:args.limit]
            texts = [to_text(space, archs[i], "code") for i in pool]
            t0 = time.time()
            feats = embed_layers(tok, model, texts, LAYERS, args.batch, name=f"{space} {name}")
            for key, f in files.items():
                np.save(f, feats[key])
            print(f"[完成] {space} {name}：{len(files)} 份特征，每份 {feats[LAYERS[0], 'mean'].shape}，"
                  f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        del model


if __name__ == "__main__":
    main()
