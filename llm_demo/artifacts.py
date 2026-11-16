"""Atomic feature writes and fail-closed provenance checks."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
ROOT=HERE.parent  # 仓库根目录；快照里的路径都相对它记录，换电脑或克隆仓库后照样能用


def repo_relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def from_repo(name):
    p = Path(name)
    return p if p.is_absolute() else ROOT / p

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def atomic_npy(path, array):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as stream:np.save(stream,array)
    temp.replace(path)

def pool_checked(pool,total):
    pool=np.asarray(pool)
    if pool.ndim!=1 or not len(pool) or pool.dtype.kind not in "iu" or len(np.unique(pool))!=len(pool):
        raise ValueError("invalid or duplicate structure pool")
    if pool.min()<0 or pool.max()>=total:raise ValueError("structure pool index out of range")
    return pool

@lru_cache(maxsize=32)
def model_fingerprint(folder):
    folder=Path(folder)
    files=[p for p in folder.iterdir() if p.suffix in {".json",".txt",".bin",".safetensors"}]
    return {p.name:file_hash(p) for p in sorted(files)}

# 文本内容已经单独记了指纹，所以这里只需要"怎么从文本算出特征"的代码
EXTRACTION_CODE = [("embed", "load_model"), ("embed", "line_spans"), ("embed", "embed_texts"),
                   ("make_bow", "bow"), ("run_layers", "embed_layers")]

def feature_context(texts, pool, model_dir, kind):
    import transformers
    import torch
    from experiment import code_digest
    sources=code_digest(EXTRACTION_CODE)
    return {"kind":kind,"texts":hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
            "pool":hashlib.sha256(np.asarray(pool).tobytes()).hexdigest(),
            "model":model_fingerprint(str(model_dir)),"sources":sources,
            "data":{str(p.resolve()):file_hash(p) for p in (HERE.parent/"predictor_lab/data").glob("*.csv")},
            "numpy":np.__version__,"transformers":transformers.__version__,"torch":torch.__version__}

def feature_cached(path,context,rows):
    path=Path(path)
    if not path.exists():return False
    meta=path.with_suffix(".manifest.json")
    if not meta.exists():
        raise ValueError(f"Legacy feature has no generation manifest: {path}. Preserve it; use a new --out/--out-dir for extraction.")
    manifest=json.loads(meta.read_text(encoding="utf8"))
    if manifest["context"]!=context or manifest["sha256"]!=file_hash(path):
        raise ValueError(f"Feature input/content changed: {path}; use a separate output directory")
    X=np.load(path,mmap_mode="r")
    if X.ndim!=2 or len(X)!=rows or not np.isfinite(X).all():raise ValueError(f"Invalid feature {path}")
    return True

def save_feature(path,array,context):
    if array.ndim!=2 or not np.isfinite(array).all():raise ValueError("invalid features")
    atomic_npy(path,array)
    meta=Path(path).with_suffix(".manifest.json");tmp=meta.with_suffix(".tmp")
    tmp.write_text(json.dumps({"context":context,"sha256":file_hash(path)},indent=2),encoding="utf8");tmp.replace(meta)

def verify_snapshot(folder):
    """Pin the audited historical files to their current data order; not a claim about their original generation."""
    folder=Path(folder);path=folder/"audit_snapshot.json"
    if not path.exists():
        pool=np.load(folder/"pool.npy")
        for f in folder.glob("*.npy"):
            if f.name=="pool.npy":continue
            side=f.with_suffix(".manifest.json")
            if not side.exists():raise ValueError(f"Untracked features: {f}")
            meta=json.loads(side.read_text(encoding="utf8"))
            if file_hash(f)!=meta["sha256"]:raise ValueError(f"Changed feature {f}")
            for data,digest in meta["context"].get("data",{}).items():
                if file_hash(data)!=digest:raise ValueError(f"Changed source data: {data}")
            if meta["context"]["pool"]!=hashlib.sha256(pool.tobytes()).hexdigest():raise ValueError("Changed pool")
            assert_rows(np.load(f,mmap_mode="r"),pool)
        return {"provenance":"generation_manifests"}
    manifest=json.loads(path.read_text(encoding="utf8"))
    listed=set()
    for name,digest in manifest["files"].items():
        p=from_repo(name)
        if not p.exists():raise ValueError(f"Audited input missing: {name}")
        if file_hash(p)!=digest:raise ValueError(f"Audited input changed: {p}")
        listed.add(p.resolve())
    # 后来新加进这个文件夹、没有登记过的特征也要拦下来
    extra=[f.name for f in folder.glob("*.npy") if f.resolve() not in listed]
    if extra:raise ValueError(f"Features not covered by {path}: {extra}")
    return manifest


def assert_rows(X,pool):
    if X.ndim!=2 or len(X)!=len(pool) or not np.isfinite(X).all():raise ValueError("feature/pool mismatch or nonfinite values")
