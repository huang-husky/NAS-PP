"""Lossless rank correction of historical trajectories; never retrain or alter tau."""
import json
import time
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from metrics import best_ranks

HERE = Path(__file__).parent
OUT = HERE / "results_repaired"
INPUT_FILES = set()

def ordinal_to_min_rank(y):
    return best_ranks(np.sort(np.asarray(y))[::-1])

def labels(space, family):
    files = {"nb201": "nasbench201.csv", "transnas": "transnas_micro.csv", "nb101": "nasbench101.csv"}
    data_path = HERE.parent / "predictor_lab" / "data" / files[space]
    INPUT_FILES.add(data_path)
    d = pd.read_csv(data_path)
    if family in {"step3", "step5", "step7", "step6"}:
        folder = HERE / "features" / ("layers" if family == "step5" else "formats") / space
    else:
        folder = HERE / "features" / ("" if space == "nb201" else space)
    INPUT_FILES.add(folder / "pool.npy")
    pool = np.load(folder / "pool.npy")
    d = d.iloc[pool]
    return {c.removesuffix("_valid") if space == "transnas" else c: d[c].to_numpy(float)
            for c in d if c.endswith("_valid") or c.endswith("_acc") or c in {"params_M", "imagenet16_acc"}}

def correct_curve(d, ys):
    d = d.copy()
    d["n_at_10_legacy"] = d.n_at_10
    for (target, n, seed), idx in d.groupby(["target", "n_train", "seed"]).groups.items():
        y = ys[target]
        te = np.random.default_rng(int(seed)).permutation(len(y))[int(n):]
        mapping = ordinal_to_min_rank(y[te])
        old = d.loc[idx, "n_at_10"].to_numpy()
        if not np.all(np.isfinite(old) & (old == np.floor(old)) & (old >= 1) & (old <= len(te))):
            raise ValueError("invalid historical rank")
        d.loc[idx, "n_at_10"] = mapping[old.astype(int)-1]
    d["metric_version"] = "min-rank-legacy-selection-v1"
    d["prediction_ties"] = "historical_numpy_argsort"
    d["diagnostics_status"] = "historical_warnings_not_recorded"
    return d

def main():
    start = time.perf_counter(); OUT.mkdir(exist_ok=True)
    changes=[]; manifest={"method":"ordinal rank -> sorted-label minimum rank; historical selection preserved", "sources":{}}
    for src in sorted((HERE / "results").rglob("*.csv")):
        if "partial" in src.name or src.name == "curves_all.csv": continue
        rel=src.relative_to(HERE / "results"); family=rel.parts[0] if len(rel.parts)>1 else "base"
        d=pd.read_csv(src); original=d.copy()
        if {"target","features","head","n_train","seed","n_at_10"}.issubset(d.columns):
            space=src.stem.removeprefix("curves_") if src.stem.startswith("curves_") else "nb201"
            d=correct_curve(d,labels(space,family)); field="n_at_10"
            assert np.array_equal(d.tau.to_numpy(),original.tau.to_numpy(),equal_nan=True)
        elif src.name == "search.csv":
            d["best_rank_legacy"]=d.best_rank
            for (space,target),idx in d.groupby(["space","target"]).groups.items():
                mapping=ordinal_to_min_rank(labels(space,"step6")[target]);old=d.loc[idx,"best_rank"].to_numpy(int)
                d.loc[idx,"best_rank"]=mapping[old-1]
            d["metric_version"]="min-rank-legacy-selection-v1"; d["prediction_ties"]="historical_numpy_argsort";field="best_rank"
            last=d[d.queries==d.queries.max()]
            summary=last.groupby(["space","target","features"]).agg(old_success=("best_rank_legacy",lambda a:(a==1).mean()),new_success=("best_rank",lambda a:(a==1).mean()),runs=("run","size"))
            (OUT/"step6").mkdir(exist_ok=True);summary.to_csv(OUT/"step6"/"success_comparison.csv")
        else:continue
        dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);d.to_csv(dest,index=False)
        changes.append({"file":str(rel),"rows":len(d),"changed":int((d[field]!=original[field]).sum()),"max_rank_correction":int((original[field]-d[field]).max())})
        manifest["sources"][str(src.resolve())]=hashlib.sha256(src.read_bytes()).hexdigest()
    pd.DataFrame(changes).to_csv(OUT/"migration_changes.csv",index=False)
    manifest["label_and_pool_inputs"]={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUT_FILES)}
    manifest["migration_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["elapsed_seconds"]=time.perf_counter()-start
    (OUT/"migration_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print(pd.DataFrame(changes).to_string(index=False));print('elapsed',manifest['elapsed_seconds'])

if __name__ == "__main__":main()
