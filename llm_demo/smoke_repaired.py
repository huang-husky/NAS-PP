"""Small actual-model check of the revised training/cache/search path."""
import time,json
from pathlib import Path
import numpy as np
import pandas as pd
import step3_formats as S
from experiment import run_curves
from step6_search import search_once
HERE=Path(__file__).parent

def main():
    start=time.perf_counter();ys,oh,cnt=S.load_space("nb201")
    feats={"onehot":oh,"code_pythia410m_step0":np.load(S.FMT_DIR/"nb201/code_pythia410m_step0.npy")}
    out=HERE/"results_repaired/smoke"
    df=run_curves("nb201",ys,list(feats),feats.__getitem__,[10,100],2,1,out,["rf"],lambda n,k:k)
    old=pd.read_csv(HERE/"results/step3/curves_nb201.csv")
    merged=df.merge(old,on=["target","features","head","n_train","seed"],suffixes=("_new","_old"),validate="one_to_one")
    if len(merged)!=8:raise AssertionError("missing smoke comparisons")
    delta=float(abs(merged.tau_new-merged.tau_old).max())
    if delta>1e-8:raise AssertionError(f"tau drift {delta}")
    cached=run_curves("nb201",ys,list(feats),feats.__getitem__,[10,100],2,1,out,["rf"],lambda n,k:k)
    pd.testing.assert_frame_equal(df,cached,check_dtype=False)
    h=search_once(oh,ys["cifar10_acc"],0,15)
    assert h[-1][0]==15
    report={"real_fits":8,"cache_reuse_verified":True,"max_tau_drift":delta,"search_budget":h[-1][0],"seconds":time.perf_counter()-start}
    (out/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf8");print(report)
if __name__=="__main__":main()
