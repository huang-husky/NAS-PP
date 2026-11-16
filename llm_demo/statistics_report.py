"""Paired effects, explicit comparison families, and stratified agreement."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t

def paired_text(df,target,head,a,b):
    sub=df[(df.target==target)&(df["head"]==head)]
    f=df.features.nunique(); family=max(1,df.target.nunique()*df["head"].nunique()*df.n_train.nunique()*f*(f-1)//2)
    out=[]
    for n,g in sub.groupby("n_train"):
        pair=g[g.features.isin([a,b])]
        if pair.duplicated(["features","seed"]).any():raise ValueError("duplicate paired seeds")
        wide=pair.pivot(index="seed",columns="features",values="tau")
        if a not in wide or b not in wide:continue
        d=(wide[a]-wide[b]).dropna().to_numpy();k=len(d)
        if k<2:continue
        p=float(wilcoxon(d).pvalue) if np.any(d) else 1.0
        half=float(t.ppf(.975,k-1)*np.std(d,ddof=1)/np.sqrt(k))
        out.append(f"n={n}: {d.mean():+.3f}, split-mean CI95=[{d.mean()-half:.3f},{d.mean()+half:.3f}], wins={sum(d>0)}/{k}, p_raw={p:.3g}, p_bonferroni={min(1,p*family):.3g} (family_bound={family})")
    return "; ".join(out)

def agreement_tables(all_df):
    points=[]
    for space,df in all_df.items():
        mean=df.groupby(["target","head","features","n_train"]).tau.mean()
        for (target,head,feat,n),y in mean.items():
            if feat in {"onehot"} or feat.endswith("_bow"):continue
            fmt,sep,model=feat.partition("_")
            if not sep:continue
            x=mean.get((target,head,fmt+"_bow",n))
            if x is not None and np.isfinite(x) and np.isfinite(y):
                points.append(dict(space=space,target=target,head=head,model=model,format=fmt,n_train=n,bow_tau=x,llm_tau=y))
    pts=pd.DataFrame(points);rows=[]
    if pts.empty:return pts,pd.DataFrame()
    for grouping in [["model","head"],["model","head","n_train"],["model","head","space","target","format","n_train"]]:
        for key,g in pts.groupby(grouping):
            r=g.bow_tau.corr(g.llm_tau) if len(g)>2 and g.bow_tau.std()>0 and g.llm_tau.std()>0 else np.nan
            rows.append(dict(zip(grouping,key if isinstance(key,tuple) else (key,)),scope="/".join(grouping),points=len(g),r=r,
                             mean_abs_gap=abs(g.llm_tau-g.bow_tau).mean(),mean_gap=(g.llm_tau-g.bow_tau).mean()))
    return pts,pd.DataFrame(rows)
