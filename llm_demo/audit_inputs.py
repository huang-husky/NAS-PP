"""Read-only raw-data crosschecks and pinned feature snapshots."""
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import file_hash, pool_checked, repo_relative
HERE=Path(__file__).parent
ROOT=HERE.parent

def main():
    report={}
    sys.path.insert(0,str(ROOT/"predictor_lab"/"tools"))
    from build_transnas import read_records,MAIN_METRIC
    d=pd.read_csv(ROOT/"predictor_lab/data/transnas_micro.csv").set_index("arch_str")
    mismatches=unfinished=0; seen=set()
    for arch,data in read_records(ROOT/"data_raw/transnas"):
        seen.add(arch)
        for task,metric in MAIN_METRIC.items():
            unfinished += not data[task]["finish_train"]
            for split in ["valid","test"]:
                expected=data[task]["metrics"][f"{split}_{metric}"][-1]
                mismatches += not np.isclose(d.loc[arch,f"{task}_{split}"],expected,rtol=0,atol=1e-10)
    report["transnas"]={"structures":len(seen),"label_mismatches":int(mismatches),"unfinished":int(unfinished)}
    db=sqlite3.connect(f"file:{ROOT/'data_raw/nasbench101/nasbench101-209f5694.db'}?mode=ro",uri=True)
    from build_nb101 import QUERY
    raw=pd.read_sql_query(QUERY,db); db.close()
    raw.columns=["hash","arch","num_vertices","test_acc","valid_acc","train_acc","params_M","n_trials"]
    d=pd.read_csv(ROOT/"predictor_lab/data/nasbench101.csv")
    merged=d.merge(raw,on="hash",suffixes=("_csv","_raw"),validate="one_to_one")
    mismatches={c:int((np.abs(merged[c+"_csv"]-merged[c+"_raw"])>0.0000500001).sum()) for c in ["test_acc","valid_acc","train_acc"]}
    report["nb101"]={"matched":len(merged),"label_mismatches_beyond_historical_rounding":mismatches,"trial_counts":sorted(raw.n_trials.unique().tolist())}
    report["nb201"]={"raw_source_available":False,"status":"raw parquet/npy absent; source axis and split semantics not independently verified"}
    if unfinished or mismatches and any(mismatches.values()) or report["transnas"]["label_mismatches"]:
        raise ValueError(report)
    datas={s:ROOT/"predictor_lab/data"/f for s,f in {"nb201":"nasbench201.csv","transnas":"transnas_micro.csv","nb101":"nasbench101.csv"}.items()}
    tables={s:pd.read_csv(p) for s,p in datas.items()}
    count=0
    for pool_file in (HERE/"features").rglob("pool.npy"):
        folder=pool_file.parent
        space=next((s for s in ["transnas","nb101"] if s in folder.parts),"nb201")
        table=tables[space];pool=pool_checked(np.load(pool_file),len(table))
        idcol={"nb201":"index","nb101":"hash","transnas":"arch_str"}[space]
        ids=table.iloc[pool][idcol].astype(str).tolist()
        if len(set(ids))!=len(ids):raise ValueError("duplicate architecture identity")
        files={repo_relative(datas[space]):file_hash(datas[space]),repo_relative(pool_file):file_hash(pool_file)}
        for f in folder.glob("*.npy"):
            if f.name=="pool.npy":continue
            a=np.load(f,mmap_mode="r")
            if a.ndim!=2 or len(a)!=len(pool) or not np.isfinite(a).all():raise ValueError(f"invalid {f}")
            files[repo_relative(f)]=file_hash(f);count+=1
        snapshot={"provenance":"audited-current-snapshot; original generation provenance not reconstructible", "space":space,"structure_ids":ids,"files":files}
        (folder/"audit_snapshot.json").write_text(json.dumps(snapshot,indent=2),encoding="utf8")
    report["feature_arrays_checked"]=count
    dest=ROOT/"docs/code_audit/input_validation.json";dest.write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
