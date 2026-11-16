"""Evaluate DeepSeek as a final selector on a random forest's NB201 shortlist."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "finetune_lab"))
from protocol import Data
from nb201 import OPS, to_arch_str

N_LABELS = 50
N_SHORTLIST = 20
N_PICK = 5
SEEDS = tuple(range(20))
API_URL = "https://api.deepseek.com/chat/completions"
MIN_GAIN = 0.05  # percentage points


def atomic_write(path, content):
    pending = path.with_suffix(".tmp")
    pending.write_text(content, encoding="utf-8")
    pending.replace(path)


def describe(data, index):
    ops = [OPS[op] for op in data.canon[index]]
    return (f"{to_arch_str(ops)} | FLOPs (M): {data.flops[index]:.8g}"
            f" | parameters (M): {data.params[index]:.8g}")


def prepare_case(data, features, seed):
    train, test = data.split(seed, N_LABELS)
    model = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed)
    predicted = model.fit(features[train], data.y[train]).predict(features[test])
    shortlist = test[np.argsort(-predicted, kind="stable")[:N_SHORTLIST]]
    shuffled = np.random.default_rng(seed + 30_000).permutation(shortlist)
    candidates = {f"C{i:02d}": index for i, index in enumerate(shuffled, 1)}

    lines = [
        "Select CNN architectures likely to give the highest CIFAR-10 test accuracy.",
        "Each cell is a directed acyclic graph with nodes 0 to 3. Node 0 is the input,",
        "node 3 the output. The three groups separated by '+' list incoming edges",
        "to nodes 1, 2 and 3, respectively. 'operation~i' means apply that operation",
        "to node i, then sum the incoming results at the destination.",
        "none removes the edge; skip_connect is identity; nor_conv_1x1 and",
        "nor_conv_3x3 are convolution blocks with batch normalization and ReLU;",
        "avg_pool_3x3 is average pooling. FLOPs and parameters describe the network.",
        "",
        f"Here are {N_LABELS} examples with measured accuracy (percent):",
        *(f"{describe(data, i)} | measured accuracy: {data.y[i]:.8g}" for i in train),
        "",
        "Candidates in shuffled order:",
        *(f"{candidate}: {describe(data, i)}" for candidate, i in candidates.items()),
        "",
        f"Choose exactly {N_PICK} distinct candidate IDs. The best measured accuracy",
        "among your choices is what matters. Return only a JSON object like",
        '{"selected":["C02","C04","C06","C08","C10"]}.',
    ]
    rf_best = float(data.y[shortlist[:N_PICK]].max())
    oracle = float(data.y[shortlist].max())
    baseline = dict(seed=seed, rf_best=rf_best, shortlist_oracle=oracle,
                    headroom=oracle - rf_best, test_oracle=float(data.y[test].max()))
    return "\n".join(lines), candidates, baseline


def selected_ids(reply, candidates):
    obj = json.loads(reply)
    ids = obj.get("selected") if isinstance(obj, dict) else None
    if (not isinstance(ids, list) or len(ids) != N_PICK
            or not all(isinstance(i, str) and i in candidates for i in ids)
            or len(set(ids)) != N_PICK):
        raise ValueError(f"Expected {N_PICK} distinct IDs from the shortlist.")
    return ids


def request_selection(prompt, candidates, model, key, cache):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = cache / f"{fingerprint}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return selected_ids(record["reply"], candidates), record, True

    import requests

    response = requests.post(API_URL, json=payload, timeout=(15, 180),
                             headers={"Authorization": f"Bearer {key}"})
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"DeepSeek finish_reason={choice.get('finish_reason')!r}")
    reply = choice["message"].get("content")
    ids = selected_ids(reply, candidates)
    record = dict(reply=reply, usage=body.get("usage", {}), model=body.get("model", model))
    atomic_write(path, json.dumps(record, indent=2))
    return ids, record, False


def save_results(rows, expected, out):
    results = pd.DataFrame(rows)
    atomic_write(out / "paired.csv", results.to_csv(index=False))
    delta = results["gain_pp"].to_numpy()
    wins = int((delta > 1e-9).sum())
    losses = int((delta < -1e-9).sum())
    complete = len(rows) == expected
    summary = dict(completed=len(rows), expected=expected,
                   mean_gain_pp=float(delta.mean()), median_gain_pp=float(np.median(delta)),
                   wins=wins, ties=len(rows) - wins - losses, losses=losses,
                   passes_pilot=(float(delta.mean()) >= MIN_GAIN and wins > losses)
                   if complete and expected == len(SEEDS) else None)
    atomic_write(out / "summary.json", json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Run the offline preview (default).")
    mode.add_argument("--smoke", action="store_true", help="One online request on a separate seed.")
    mode.add_argument("--run", action="store_true", help="Run the 20-seed online experiment.")
    parser.add_argument("--model", choices=("deepseek-v4-pro", "deepseek-flash"),
                        default="deepseek-v4-pro")
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    online = args.smoke or args.run
    if online and not key:
        parser.error("Set DEEPSEEK_API_KEY in your terminal before enabling API calls.")

    stage = "pilot" if args.run else "smoke" if args.smoke else "preview"
    seeds = (100,) if args.smoke else SEEDS
    out = HERE / "results" / "rerank" / args.model / stage
    out.mkdir(parents=True, exist_ok=True)
    data = Data()
    features = data.feature_sets()["canon+fp"]
    cases = [prepare_case(data, features, seed) for seed in seeds]
    baselines = pd.DataFrame([baseline for _, _, baseline in cases])
    atomic_write(out / "baseline.csv", baselines.to_csv(index=False))
    atomic_write(out / "prompt_preview.txt", cases[0][0] + "\n")
    print(f"RF already selects a shortlist optimum on "
          f"{int((baselines.headroom < 1e-9).sum())}/{len(seeds)} seeds. "
          f"Mean possible gain: {baselines.headroom.mean():.3f} pp.", flush=True)
    if not online:
        print(f"Offline preview saved to {out}. No API requests.", flush=True)
        return

    cache = out.parent / "cache"
    cache.mkdir(exist_ok=True)
    rows = []
    for prompt, candidates, baseline in cases:
        seed = baseline["seed"]
        try:
            ids, record, cached = request_selection(prompt, candidates, args.model, key, cache)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            raise SystemExit(f"Seed {seed} failed: {error}. "
                             "Re-run this command to reuse completed replies.") from None
        best = float(max(data.y[candidates[candidate]] for candidate in ids))
        row = dict(baseline, deepseek_best=best, gain_pp=best - baseline["rf_best"],
                   deepseek_regret=baseline["test_oracle"] - best,
                   rf_regret=baseline["test_oracle"] - baseline["rf_best"],
                   selected=" ".join(ids), response_model=record["model"],
                   prompt_tokens=record["usage"].get("prompt_tokens"),
                   completion_tokens=record["usage"].get("completion_tokens"))
        rows.append(row)
        summary = save_results(rows, len(cases), out)
        print(f"[{len(rows)}/{len(cases)}] seed={seed} gain={row['gain_pp']:+.3f} pp"
              f"{' (cached)' if cached else ''}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
