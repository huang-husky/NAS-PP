"""规律 2：同一批结构、五种写法，分别提取 LLM 特征和词频特征。

每个空间用一批固定的结构（结构池），存到 features/formats/<空间>/:
    nb201      从 15625 个里随机抽 5000 个（种子 2026）
    transnas   全部 4096 个
    nb101      沿用 features/nb101/pool.npy 那 5000 个
文件:
    pool.npy                    结构池（在原数据里的编号）
    <写法>_bow.npy              词频特征（Pythia 分词器，去掉所有结构都一样的 token）
    <写法>_<模型>.npy           LLM 平均特征
code 写法的 LLM 特征第一阶段已经算过，直接从原来的文件里按结构池取出来，不重算。

运行方式（在 llm_demo 文件夹下）:
    python run_formats.py --limit 32 --out features/_smoke_formats    冒烟测试
    python run_formats.py                                             正式运行（先随机 Pythia，再训练完成的 Pythia）
已经算好的文件会自动跳过，中途停了重新运行就能接着跑。
"""

import argparse
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from embed import FEATURE_DIR, embed_texts, load_archs, load_model
from formats import FORMATS, to_text
from make_bow import TOKENIZER, bow
from models import MODEL_DIR

SPACES = ["nb201", "transnas", "nb101"]
MODELS = ["pythia410m_step0", "pythia410m_step143000"]
OLD_DIR = {"nb201": FEATURE_DIR, "transnas": FEATURE_DIR / "transnas", "nb101": FEATURE_DIR / "nb101"}


def get_pool(space, out_dir):
    f = out_dir / "pool.npy"
    if f.exists():
        return np.load(f)
    if space == "nb201":
        pool = np.sort(np.random.default_rng(2026).choice(15625, size=5000, replace=False))
    else:
        pool = np.load(OLD_DIR[space] / "pool.npy")
    np.save(f, pool)
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default=",".join(SPACES))
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="只用结构池里的前几个（冒烟测试用）")
    parser.add_argument("--out", default=str(FEATURE_DIR / "formats"))
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    spaces, formats, models = args.spaces.split(","), args.formats.split(","), args.models.split(",")

    # ---- 先把所有文本和词频准备好（几分钟），顺便打印每种写法的例子
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    texts, pools = {}, {}
    for space in spaces:
        out_dir = Path(args.out) / space
        out_dir.mkdir(parents=True, exist_ok=True)
        archs = load_archs(space)
        pool = get_pool(space, out_dir)
        if args.limit:
            pool = pool[:args.limit]
        pools[space] = pool
        for fmt in formats:
            texts[space, fmt] = [to_text(space, archs[i], fmt) for i in pool]
            bow_file = out_dir / f"{fmt}_bow.npy"
            if not bow_file.exists():
                X, words = bow(texts[space, fmt], tokenizer)
                np.save(bow_file, X)
                print(f"[词频] {space:8s} {fmt:8s} {X.shape[1]:3d} 种会变的 token："
                      f"{' '.join(repr(w) for w in words[:20])}{' ……' if len(words) > 20 else ''}", flush=True)

    # ---- LLM 特征：每个模型只加载一次，跑完所有空间和写法
    total_left = sum(len(pools[s]) for s in spaces for f in formats if f != "code") * len(models)
    print(f"\n一共要编码约 {total_left} 段文本（已存在的会跳过）\n", flush=True)
    for name in models:
        model = tok = None
        for space in spaces:
            out_dir = Path(args.out) / space
            for fmt in formats:
                out_file = out_dir / f"{fmt}_{name}.npy"
                if out_file.exists():
                    print(f"[跳过] {space}/{out_file.name} 已存在", flush=True)
                    continue
                if fmt == "code" and not args.limit and (OLD_DIR[space] / f"{name}.npy").exists():  # 第一阶段算过的，按结构池取出来
                    old = np.load(OLD_DIR[space] / f"{name}.npy")
                    old_pool = np.load(OLD_DIR[space] / "pool.npy")
                    where = {p: k for k, p in enumerate(old_pool)}
                    np.save(out_file, old[[where[p] for p in pools[space]]])
                    print(f"[复用] {space}/{out_file.name}（第一阶段的 code 特征）", flush=True)
                    continue
                if model is None:
                    tok, model = load_model(MODEL_DIR / name)
                t0 = time.time()
                means, _ = embed_texts(tok, model, texts[space, fmt], args.batch,
                                       name=f"{space} {fmt} {name}", per_line=False)
                np.save(out_file, means)
                print(f"[完成] {space}/{out_file.name} {means.shape}  用时 {(time.time() - t0) / 60:.1f} 分钟",
                      flush=True)
        del model


if __name__ == "__main__":
    main()
