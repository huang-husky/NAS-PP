"""Small regression tests; extract search functions to avoid loading LLM dependencies."""
import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from metrics import best_ranks, top_indices


def load_functions(file, names):
    tree = ast.parse(Path(__file__).with_name(file).read_text(encoding="utf8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"np": np, "pd": pd, "best_ranks": best_ranks, "top_indices": top_indices}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, "exec"), env)
    return env

search = load_functions("step6_search.py", {"search_once", "complete_search_rows"})
metrics = load_functions("evaluate.py", {"n_at_k"})

class MetricFixes(unittest.TestCase):
    def test_equal_best(self):
        self.assertEqual(best_ranks([1., 1., 0.]).tolist(), [1, 1, 3])
        self.assertEqual(metrics["n_at_k"]([0., 1., 0.], [1., 1., 0.], 1), 1)

    def test_stable_prediction_ties(self):
        self.assertEqual(top_indices([1., 1., 0.], 2).tolist(), [0, 1])

    def test_budget(self):
        for budget in [1, 5, 10, 15, 20, 25]:
            h = search["search_once"](None, np.arange(25.), 0, budget)
            self.assertEqual(h[-1][0], budget)
            self.assertTrue(all(q <= budget for q, r in h))
        for budget in [0, 26]:
            with self.assertRaises(ValueError):
                search["search_once"](None, np.arange(25.), 0, budget)

    def test_cache_completion(self):
        def frame(q):
            return pd.DataFrame([dict(space="s", target="t", features="f", run=0,
                                      queries=x, best_rank=1) for x in q])
        complete = search["complete_search_rows"]
        self.assertTrue(complete(frame([10, 20, 40]), 40).empty)
        self.assertTrue(complete(frame([10, 20, 30, 40]), 200).empty)
        self.assertTrue(complete(frame([10, 10, 20]), 20).empty)
        self.assertEqual(len(complete(frame([10, 15]), 15)), 2)

if __name__ == "__main__":
    unittest.main()
