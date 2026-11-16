"""Recover undefined constant-feature tau; replay eight ambiguous NB101 fits."""
import time,json,warnings,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from evaluate import make_head
from metrics import best_ranks
from migrate_metrics import labels
HERE=Path(__file__).parent

def main():
    start=time.perf_counter();report=[]
    for space in ["nb201","transnas","nb101"]:
        path=HERE/"results_repaired/step5"/f"curves_{space}.csv"
        old=pd.read_csv(HERE/"results/step5"/path.name);d=pd.read_csv(path)
        d["tau_legacy"]=old.tau
        d["tau_status"]="historical_finite"
        ys=labels(space,"step5")
        for feat,idx in old[old.tau==0].groupby("features").groups.items():
            X=np.load(HERE/"features/layers"/space/(feat+".npy"))
            constant=np.array_equal(X,np.broadcast_to(X[:1],X.shape))
            for i in idx:
                r=old.loc[i]
                if constant:
                    d.loc[i,"tau"]=np.nan;d.loc[i,"tau_status"]="undefined_constant_features_verified"
                    continue
                y=ys[r.target];perm=np.random.default_rng(int(r.seed)).permutation(len(y));tr,te=perm[:int(r.n_train)],perm[int(r.n_train):]
                model=make_head(r["head"],int(r.seed))
                if r["head"]=="rf":model.set_params(regressor__randomforestregressor__n_jobs=1)
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    pred=model.fit(X[tr],y[tr]).predict(X[te]);tau=kendalltau(pred,y[te]).statistic
                # Preserve historical selection convention, as in the metric migration.
                n10=int(best_ranks(y[te])[np.argsort(-pred)[:10]].min())
                if n10 != d.loc[i,"n_at_10"]:raise AssertionError("historical replay N@10 differs")
                d.loc[i,"tau"]=tau;d.loc[i,"tau_status"]="undefined_replayed" if np.isnan(tau) else "finite_replayed"
                report.append(dict(space=space,feature=feat,target=r.target,n_train=int(r.n_train),seed=int(r.seed),head=r["head"],tau=None if np.isnan(tau) else float(tau),warnings=[str(w.message) for w in seen]))
        d.to_csv(path,index=False)
    result={"exact_constant_rows":720,"replayed_rows":len(report),"seconds":time.perf_counter()-start,"replays":report,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/"results_repaired/undefined_tau_repair.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
