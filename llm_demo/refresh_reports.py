"""Recompute references, statistics and figures without retraining models."""
from pathlib import Path
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import step3_formats as S
from statistics_report import agreement_tables, paired_text
from explain import group_mean_reference
from step2 import load_space, encode_any
from step6_search import plot_search
from experiment import atomic_csv
HERE=Path(__file__).parent
OUT=HERE/"results_repaired"

def main():
    t=time.perf_counter()
    dfs={s:pd.read_csv(OUT/"step3"/f"curves_{s}.csv") for s in S.TARGETS}
    points,stats=agreement_tables(dfs)
    atomic_csv(points,OUT/"step3/agreement_points.csv")
    atomic_csv(stats,OUT/"step3/agreement_stratified.csv")
    references=[]
    for space in S.TARGETS:
        archs,ys,ops=load_space(space);_,cnt=encode_any(space,archs,ops)
        for target,y in ys.items():
            value,groups=group_mean_reference(cnt,y)
            references.append(dict(space=space,target=target,n_arch=len(y),groups=groups,reference=value,definition="in-sample group-mean oracle, not a bound"))
    atomic_csv(pd.DataFrame(references),OUT/"group_mean_references.csv")
    S.set_models(list(dict.fromkeys(points.model)))
    S.plot_scatter(dfs,OUT/"step3")
    S.plot_curves(dfs,OUT/"step3")
    import make_summary_fig
    make_summary_fig.main()
    make_summary_fig.main(counts_baseline=True)
    plot_search(pd.read_csv(OUT/"step6/search.csv"),OUT/"step6")
    effects=[]
    for space,d in dfs.items():
        for (target,head),g in d.groupby(["target","head"]):
            for model in ["pythia410m_step0","pythia410m_step143000","qwen25_0.5b"]:
                for fmt in S.FORMATS:
                    effects.append(dict(space=space,target=target,head=head,model=model,format=fmt,
                        paired_effect=paired_text(d,target,head,f"{fmt}_{model}",f"{fmt}_bow")))
    atomic_csv(pd.DataFrame(effects),OUT/"step3/paired_effects.csv")
    from experiment import write_tau_summary
    from step5_layers import plot as plot_layers
    layer_dfs={s:pd.read_csv(OUT/"step5"/f"curves_{s}.csv") for s in S.TARGETS}
    plot_layers(layer_dfs,OUT/"step5")
    for path in OUT.rglob("*.csv"):
        if (path.name.startswith("curves_") and not path.name.endswith("_quality.csv")) or path.name in {"curves.csv","results.csv"}:
            data=pd.read_csv(path)
            if {"tau","target","head","features","n_train"}.issubset(data.columns):
                write_tau_summary(data,path.with_name(path.stem+"_quality.csv"))
    print('refresh seconds',time.perf_counter()-t)
if __name__=="__main__":main()
