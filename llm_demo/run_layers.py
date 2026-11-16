"""规律 4：取特征的方式会不会改变"冻结特征 ≈ 词频"？

之前的特征都是"最后一层、所有 token 取平均"。这里换两个维度:
    哪一层     第 0 层（词嵌入，还没经过任何注意力）、第 6、12、18、24 层（410M 一共 24 层）
    怎么取     mean：所有 token 取平均；last：只取最后一个 token
待检验的经验问题:
    第0层mean是token嵌入的长度归一化和；不保证与原始计数有相同的回归性能。
    后续层已经包含上下文信息，平均池化不保证消除位置信息。
    比较mean与last的实际预测表现，不把词频参考当作LLM上界。

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
from artifacts import feature_context, feature_cached, save_feature, atomic_npy, pool_checked

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
    global FMT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(FEATURE_DIR / "layers"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--source-dir", default=str(FMT_DIR))
    args = parser.parse_args()
    FMT_DIR = Path(args.source_dir)

    if args.limit is not None and (args.limit < 1 or Path(args.out).resolve() == (FEATURE_DIR / "layers").resolve()):
        raise ValueError("--limit requires positive size and a separate --out")
    for name in args.models.split(","):
        tok, model = load_model(MODEL_DIR / name)
        for space in SPACES:
            out_dir = Path(args.out) / space
            out_dir.mkdir(parents=True, exist_ok=True)
            original_pool = np.load(FMT_DIR / space / "pool.npy")
            wanted_pool = original_pool[:args.limit]
            pool_file = out_dir / "pool.npy"
            if pool_file.exists() and not np.array_equal(np.load(pool_file), wanted_pool):
                raise ValueError("Existing layer pool differs; use a new output directory")
            if not pool_file.exists():
                atomic_npy(pool_file, wanted_pool)
            files = {(l, p): out_dir / f"{name}_L{l}_{p}.npy" for l in LAYERS for p in ["mean", "last"]}
            archs = load_archs(space)
            pool = pool_checked(np.load(out_dir / "pool.npy"), len(archs))
            texts = [to_text(space, archs[i], "code") for i in pool]
            from make_bow import TOKENIZER
            from artifacts import file_hash
            bow_context = feature_context(texts, pool, TOKENIZER, "copied-bow:" + file_hash(FMT_DIR / space / "code_bow.npy"))
            if not feature_cached(out_dir / "code_bow.npy", bow_context, len(pool)):
                save_feature(out_dir / "code_bow.npy", np.load(FMT_DIR / space / "code_bow.npy")[:args.limit], bow_context)
            context = feature_context(texts, pool, MODEL_DIR / name, "layers:" + repr(LAYERS))
            ready = [feature_cached(f, context, len(pool)) for f in files.values()]
            if all(ready):
                continue
            t0 = time.time()
            feats = embed_layers(tok, model, texts, LAYERS, args.batch, name=f"{space} {name}")
            for key, f in files.items():
                save_feature(f, feats[key], context)
            print(f"[完成] {space} {name}：{len(files)} 份特征，每份 {feats[LAYERS[0], 'mean'].shape}，"
                  f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        del model


if __name__ == "__main__":
    main()
