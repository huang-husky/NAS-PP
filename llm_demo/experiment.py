"""Validated, versioned experiment execution shared across learning-curve scripts."""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy.stats import kendalltau
from joblib import Parallel, delayed
from metrics import METRIC_VERSION, best_ranks, top_indices

HERE = Path(__file__).parent
KEYS = ["target", "features", "head", "n_train", "seed"]

def atomic_csv(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False); temp.replace(path)

def array_digest(a):
    a=np.asarray(a)
    h=hashlib.sha256(str((a.shape,str(a.dtype))).encode())
    h.update(np.ascontiguousarray(a).tobytes()); return h.hexdigest()

# 只有这些函数会影响一次训练和打分的结果；改别的文件（注释、画图、测试）不让缓存失效
COMPUTING_CODE = [("evaluate", "make_head"), ("experiment", "validate_xy"), ("experiment", "evaluate_once"),
                  ("metrics", "best_ranks"), ("metrics", "top_indices")]

def code_digest(functions):
    import importlib, inspect
    h=hashlib.sha256()
    for module, name in functions:
        h.update(f"{module}.{name}".encode())
        h.update(inspect.getsource(getattr(importlib.import_module(module), name)).encode())
    return h.hexdigest()

def source_digest():
    return code_digest(COMPUTING_CODE)

def validate_xy(X,y,n=None):
    X=np.asarray(X); y=np.asarray(y)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or not len(y):
        raise ValueError("feature/label dimensions do not match")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite feature or label")
    if n is not None and (not isinstance(n,(int,np.integer)) or not 1<=n<len(y)):
        raise ValueError("training size must leave a nonempty test set")

def evaluate_once(X,y,n,seed,head):
    from evaluate import make_head
    validate_xy(X,y,n)
    perm=np.random.default_rng(seed).permutation(len(y)); tr,te=perm[:n],perm[n:]
    model=make_head(head,seed)
    if head=="rf": model.set_params(regressor__randomforestregressor__n_jobs=1)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        pred=model.fit(X[tr],y[tr]).predict(X[te])
        if not np.isfinite(pred).all(): raise ValueError("nonfinite model predictions")
        raw=float(kendalltau(pred,y[te]).statistic)
    status="ok" if np.isfinite(raw) else "undefined_constant_ranking"
    return dict(tau=raw, tau_raw=raw, tau_status=status,
                n_at_10=int(best_ranks(y[te])[top_indices(pred,min(10,len(te)))].min()),
                warnings=json.dumps([f"{w.category.__name__}: {w.message}" for w in seen],ensure_ascii=False),
                metric_version=METRIC_VERSION, prediction_ties="stable_input_order")

def run_curves(space,ys,features,load,sizes,n_seeds,n_jobs,out_dir,heads,seeds_for):
    """Cache only records matching inputs, code, dependencies, metric and exact seeds."""
    if n_seeds<1 or not sizes: raise ValueError("positive seeds and nonempty sizes required")
    out_dir=Path(out_dir); rows=[]; source=source_digest()
    with Parallel(n_jobs=n_jobs) as workers:
        for feature in features:
            X=load(feature); xd=array_digest(X)
            for target,y in ys.items():
                validate_xy(X,y)
                for head in heads:
                    for n in sizes:
                        validate_xy(X,y,n); k=seeds_for(n,n_seeds)
                        manifest=dict(space=space,features=feature,target=target,head=head,n_train=n,
                                      X=xd,y=array_digest(y),source=source,metric=METRIC_VERSION,
                                      numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
                        key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
                        path=out_dir/"cache_v3"/(key+".csv")
                        prev=pd.read_csv(path) if path.exists() else pd.DataFrame()
                        have=set()
                        if not prev.empty:
                            required={*KEYS,"tau","tau_raw","tau_status","warnings","n_at_10","metric_version","cache_key"}
                            if not required.issubset(prev.columns) or prev.duplicated(KEYS).any():
                                raise ValueError(f"invalid cache {path}")
                            if not (prev.cache_key==key).all() or not (prev.metric_version==METRIC_VERSION).all():
                                raise ValueError(f"cache signature mismatch {path}")
                            if not (prev.seed>=0).all() or not (prev.seed==np.floor(prev.seed)).all():
                                raise ValueError(f"invalid seed {path}")
                            if not prev.tau_status.isin(["ok","undefined_constant_ranking"]).all():
                                raise ValueError(f"invalid status {path}")
                            good=prev.tau_status=="ok"
                            if not np.isfinite(prev.loc[good,"tau"]).all() or (prev.loc[good,"tau"].abs()>1).any():
                                raise ValueError(f"invalid tau {path}")
                            if not np.isfinite(prev.n_at_10).all() or not prev.n_at_10.between(1,len(y)-n).all():
                                raise ValueError(f"invalid rank {path}")
                            have=set(prev.seed.astype(int))
                        todo=[s for s in range(k) if s not in have]
                        new=workers(delayed(evaluate_once)(X,y,n,s,head) for s in todo)
                        fresh=pd.DataFrame([dict(target=target,features=feature,head=head,n_train=n,seed=s,
                                                  cache_key=key,**r) for s,r in zip(todo,new)])
                        combined=pd.concat([prev,fresh],ignore_index=True)
                        if todo:
                            atomic_csv(combined,path)
                            path.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
                        rows.append(combined[combined.seed.isin(range(k))])
                        print(f"[{space}] {feature} {target} {head} n={n}: new={len(todo)}, cached={k-len(todo)}",flush=True)
    df=pd.concat(rows,ignore_index=True)
    if df.duplicated(KEYS).any(): raise ValueError("duplicate experimental keys")
    atomic_csv(df,out_dir/f"curves_{space}_partial.csv")
    write_tau_summary(df, out_dir / f"quality_{space}.csv")
    return df


def write_tau_summary(df, path):
    summary=df.groupby(["target","head","features","n_train"]).tau.agg(
        mean="mean",std="std",n_valid="count",n_total="size")
    summary["n_undefined"]=summary.n_total-summary.n_valid
    atomic_csv(summary.reset_index(),path)
