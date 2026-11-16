import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from artifacts import atomic_npy, feature_cached, save_feature, pool_checked
from experiment import run_curves, evaluate_once, validate_xy
from migrate_metrics import ordinal_to_min_rank, correct_curve
from statistics_report import paired_text, agreement_tables
from explain import group_mean_reference

class PipelineTests(unittest.TestCase):
    def test_lossless_rank_conversion(self):
        rng=np.random.default_rng(3)
        for _ in range(1000):
            y=rng.integers(0,8,30);selected=rng.choice(30,10,replace=False)
            old=(np.argsort(np.argsort(-y))+1)[selected].min()
            self.assertEqual(ordinal_to_min_rank(y)[old-1],rankdata(-y,method="min")[selected].min())

    def test_migration_preserves_tau(self):
        y=np.array([3,3,3,2,2,1.]);d=pd.DataFrame([dict(target="t",n_train=1,seed=0,n_at_10=2,tau=.5)])
        fixed=correct_curve(d,{"t":y})
        self.assertEqual(fixed.tau.iloc[0],.5);self.assertEqual(fixed.n_at_10.iloc[0],1)

    def test_reference_is_not_bound(self):
        y=np.array([0.,1.,100.,2.,3.,4.]);x=np.array([[0]]*3+[[1]]*3)
        self.assertLess(group_mean_reference(x,y)[0],0)

    def test_pool_validation(self):
        for p in [np.array([0,0]),np.array([0,3]),np.array([.2,1.])]:
            with self.assertRaises(ValueError):pool_checked(p,3)

    def test_feature_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.npy";a=np.ones((3,2));context={"test":1}
            save_feature(p,a,context)
            self.assertTrue(feature_cached(p,context,3))
            with self.assertRaises(ValueError):feature_cached(p,{"test":2},3)
            with self.assertRaises(ValueError):feature_cached(p,context,2)
            atomic_npy(p,np.zeros((3,2)))
            with self.assertRaises(ValueError):feature_cached(p,context,3)

    def test_xy_validation(self):
        with self.assertRaises(ValueError):validate_xy(np.ones((3,2)),np.ones(4),1)
        with self.assertRaises(ValueError):validate_xy(np.ones((3,2)),np.ones(3),3)
        with self.assertRaises(ValueError):validate_xy(np.ones((3,2)),np.array([1,2,np.nan]),1)

    def test_real_constant_model_keeps_undefined_tau(self):
        r=evaluate_once(np.ones((20,2)),np.ones(20),10,0,"rf")
        self.assertTrue(np.isnan(r["tau"]));self.assertEqual(r["tau_status"],"undefined_constant_ranking")
        self.assertEqual(r["n_at_10"],1)

    def test_cache_exact_seeds_and_input_version(self):
        calls=[]
        def fake(X,y,n,seed,head):
            calls.append(seed)
            return dict(tau=.5,tau_raw=.5,tau_status="ok",warnings="[]",n_at_10=1,
                        metric_version="min-rank-stable-ties-v2",prediction_ties="stable_input_order")
        with tempfile.TemporaryDirectory() as td,patch("experiment.evaluate_once",fake):
            X=np.arange(60.).reshape(20,3);y=np.arange(20.)
            def run(k):return run_curves("s",{"t":y},["f"],lambda f:X,[10],k,1,td,["rf"],lambda n,k:k)
            run(2);self.assertEqual(calls,[0,1]);run(2);self.assertEqual(calls,[0,1])
            run(3);self.assertEqual(calls,[0,1,2]);self.assertEqual(len(run(1)),1)
            y[0]=99;run(1);self.assertEqual(calls,[0,1,2,0])

    def test_pairing_requires_seed_intersection(self):
        d=pd.DataFrame([dict(target="t",head="rf",features=f,n_train=10,seed=i,tau=i*.1) for f,ids in [("a",[0,1]),("b",[2,3])] for i in ids])
        self.assertEqual(paired_text(d,"t","rf","a","b"),"")

if __name__=="__main__":unittest.main()
