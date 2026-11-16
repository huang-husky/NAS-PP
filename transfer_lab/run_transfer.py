"""Paired NB101/NB201 transfer experiment. See README.md for the protocol."""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "finetune_lab"))
sys.path.insert(0, str(ROOT / "llm_demo"))
from protocol import Data, EDGES, OPS, evaluate
from nb101_text import NB101_CSV

MODEL = ROOT / "llm_demo" / "models" / "pythia70m_step143000"
SPACES = ("nb101", "nb201")
METHODS = ("pythia", "gnn")
SETTINGS = ("target_only", "source_then_target")
NB101_OPS = {"conv1x1-bn-relu": "conv1x1", "conv3x3-bn-relu": "conv3x3",
             "maxpool3x3": "maxpool3x3"}
NB201_OPS = {"skip_connect": "identity", "nor_conv_1x1": "conv1x1",
             "nor_conv_3x3": "conv3x3", "avg_pool_3x3": "avgpool3x3"}
NODE_TYPES = ("input", "sum", "output_sum", "output_concat",
              *NB101_OPS.values(), "identity", "avgpool3x3")
MAX_NODES = 10  # NB201: four states plus up to six operation nodes
KEY = ["source", "target", "seed", "n_target", "model", "setting"]


def graph101(arch):
    last = sum(key.startswith("input") for key in arch)
    nodes = ["input"] + [NB101_OPS[arch[f"op{i}"]] for i in range(1, last)] + ["output_concat"]
    edges = [(src, dst) for dst in range(1, last + 1) for src in arch[f"input{dst}"]]
    return nodes, sorted(edges)


def graph201(row):
    nodes = ["input", "sum", "sum", "output_sum"]
    edges = []
    for (src, dst), op in zip(EDGES, row):
        if OPS[op] != "none":
            index = len(nodes)
            nodes.append(NB201_OPS[OPS[op]])
            edges.extend([(src, index), (index, dst)])
    active = sorted({0, 3} | {node for edge in edges for node in edge})
    remap = {old: new for new, old in enumerate(active)}
    return [nodes[i] for i in active], sorted((remap[u], remap[v]) for u, v in edges)


@dataclass
class Examples:
    inputs: dict
    labels: np.ndarray


def prepare(graphs, labels, tokenizer):
    features = torch.zeros(len(graphs), MAX_NODES, len(NODE_TYPES))
    adjacency = torch.zeros(len(graphs), MAX_NODES, MAX_NODES)
    texts = []
    for i, (nodes, edges) in enumerate(graphs):
        features[i, range(len(nodes)), [NODE_TYPES.index(node) for node in nodes]] = 1
        for src, dst in edges:
            adjacency[i, dst, src] = 1
        if tokenizer is not None:
            node_text = ", ".join(f"{j}={node}" for j, node in enumerate(nodes))
            edge_text = ", ".join(f"{src}->{dst}" for src, dst in edges)
            texts.append(f"Nodes: {node_text}. Edges: {edge_text}.")
    inputs = {"gnn": (features, adjacency)}
    if tokenizer is not None:
        tokens = tokenizer(texts, padding=True, return_tensors="pt")
        inputs["pythia"] = (tokens.input_ids, tokens.attention_mask)
    return Examples(inputs, np.asarray(labels, dtype=float))


def select_examples(nb101, nb201, space, seed, n_train, n_test, tokenizer):
    if space == "nb201":
        train, test = nb201.split(seed, n_train)
        indices = np.concatenate([train, test[:n_test]])
        graphs = [graph201(nb201.canon[i]) for i in indices]
        labels = nb201.y[indices]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(nb101), n_train + n_test, replace=False, shuffle=True)
        graphs = [graph101(json.loads(arch)) for arch in nb101.arch.iloc[indices]]
        labels = nb101.test_acc.to_numpy()[indices]
    return prepare(graphs, labels, tokenizer)


class GraphRanker(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.project = nn.Linear(len(NODE_TYPES), width)
        self.layers = nn.ModuleList(nn.Linear(3 * width, width) for _ in range(3))
        self.head = nn.Sequential(nn.Linear(width + len(NODE_TYPES), width),
                                  nn.ReLU(), nn.Linear(width, 1))

    def forward(self, features, incoming):
        mask = features.sum(-1, keepdim=True)
        outgoing = incoming.transpose(1, 2)
        incoming = incoming / incoming.sum(-1, keepdim=True).clamp_min(1)
        outgoing = outgoing / outgoing.sum(-1, keepdim=True).clamp_min(1)
        hidden = self.project(features).relu() * mask
        for layer in self.layers:
            signals = torch.cat([hidden, incoming @ hidden, outgoing @ hidden], dim=-1)
            hidden = (hidden + layer(signals).relu()) * mask
        pooled = hidden.sum(1) / mask.sum(1)
        return self.head(torch.cat([pooled, features.sum(1)], dim=-1)).squeeze(-1)


class PythiaRanker(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel

        self.lm = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        width = self.lm.config.hidden_size
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, attention):
        # Trim padding shared by the entire batch.
        length = int(attention.sum(1).max())
        mask = attention[:, :length]
        hidden = self.lm(input_ids=ids[:, :length], attention_mask=mask,
                         use_cache=False).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(self.norm(pooled)).squeeze(-1)


def fit(model, inputs, labels, steps, batch_size, lr, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if np.unique(labels).size < 2:
        raise ValueError("The selected training labels have no ranking pairs.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(steps):
        while True:
            sample = rng.choice(len(labels), min(batch_size, len(labels)), replace=False)
            truth = labels[sample]
            pairs = truth[:, None] > truth[None, :]
            if pairs.any():
                break
        scores = model(*(tensor[sample].to(device) for tensor in inputs))
        better = torch.from_numpy(pairs).to(device)
        loss = F.softplus(-(scores[:, None] - scores[None, :])[better]).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step + 1 in (1, steps) or (step + 1) % 100 == 0:
            print(f"    {step + 1}/{steps} steps, loss={loss.item():.4f}", flush=True)


@torch.inference_mode()
def predict(model, inputs, start, batch_size, device):
    model.eval()
    predictions = []
    for offset in range(start, len(inputs[0]), batch_size):
        predictions.append(model(*(tensor[offset:offset + batch_size].to(device)
                                   for tensor in inputs)).cpu().numpy())
    scores = np.concatenate(predictions)
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores.")
    return scores


def save_csv(frame, path):
    pending = path.with_suffix(".tmp")
    frame.to_csv(pending, index=False)
    pending.replace(path)


def report(rows, out):
    results = pd.DataFrame(rows).sort_values(KEY)
    save_csv(results, out / "runs.csv")
    metrics = ["tau", "rho", "regret10"]
    means = results.groupby([key for key in KEY if key != "seed"])[metrics]
    save_csv(means.agg(["mean", "std", "count"]).round(6).set_axis(
        [f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std", "count")],
        axis=1).reset_index(), out / "summary.csv")
    paired = results.pivot(index=KEY[:4], columns=KEY[4:], values=metrics)
    deltas = {}
    for metric in metrics:
        for model in METHODS:
            transfer = (metric, model, "source_then_target")
            target = (metric, model, "target_only")
            if transfer in paired and target in paired:
                deltas[f"{metric}_{model}_transfer_gain"] = paired[transfer] - paired[target]
        pythia, gnn = (metric, "pythia", "source_then_target"), (metric, "gnn", "source_then_target")
        if pythia in paired and gnn in paired:
            deltas[f"{metric}_pythia_minus_gnn_after_transfer"] = paired[pythia] - paired[gnn]
        pythia_gain, gnn_gain = (f"{metric}_{model}_transfer_gain" for model in METHODS)
        if pythia_gain in deltas and gnn_gain in deltas:
            deltas[f"{metric}_extra_transfer_gain"] = deltas[pythia_gain] - deltas[gnn_gain]
    save_csv(pd.DataFrame(deltas, index=paired.index).reset_index(), out / "paired.csv")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--gnn-only", action="store_true")
    parser.add_argument("--pythia-lr", type=float, default=5e-5)
    parser.add_argument("--gnn-lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.threads < 1 or not all(np.isfinite(lr) and lr > 0 for lr in (args.pythia_lr, args.gnn_lr)):
        parser.error("Threads and learning rates must be positive.")
    return args


def main():
    args = arguments()
    n_source, n_test, budgets, seeds, source_steps, target_steps = {
        "smoke": (32, 24, [10], [100], 2, 2),
        "pilot": (1000, 2000, [50], [100, 101, 102], 300, 150),
        "full": (1000, 2000, [10, 50, 100], list(range(5)), 300, 150),
    }[args.preset]
    methods = ("gnn",) if args.gnn_only else METHODS
    out = args.out or (HERE / "results" / (args.preset + ("_gnn" if args.gnn_only else "")))
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the selected PyTorch environment.")
    if not args.gnn_only and not (args.model_path / "config.json").is_file():
        raise SystemExit(f"Local Pythia model not found: {args.model_path}")

    config = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preset=args.preset, methods=list(methods), n_source=n_source, n_test=n_test,
                  budgets=budgets, seeds=seeds, source_steps=source_steps, target_steps=target_steps,
                  batch_size=16, pythia_lr=args.pythia_lr, gnn_lr=args.gnn_lr,
                  device=args.device, threads=args.threads, model_path=str(args.model_path.resolve()))
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "config.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise SystemExit("Output directory contains another configuration. Choose a new --out.")
    manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    results_file = out / "runs.csv"
    rows = pd.read_csv(results_file).to_dict("records") if results_file.exists() else []
    done = {tuple(row[key] for key in KEY) for row in rows}
    if rows:
        report(rows, out)
    if len(done) == len(SPACES) * len(seeds) * len(budgets) * len(methods) * len(SETTINGS):
        print(f"All runs already complete: {out.resolve()}", flush=True)
        return

    print(f"Preparing local datasets and models ({args.preset}, {device})", flush=True)
    tokenizer = None
    if not args.gnn_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    nb101 = pd.read_csv(NB101_CSV, usecols=["arch", "test_acc"])
    nb201 = Data()
    for source, target in (SPACES, SPACES[::-1]):
        for seed in seeds:
            jobs = [(model, setting, n) for model in methods for setting in SETTINGS for n in budgets
                    if (source, target, seed, n, model, setting) not in done]
            if not jobs:
                continue
            source_data = select_examples(nb101, nb201, source, seed + 20_000, n_source, 0, tokenizer)
            target_data = select_examples(nb101, nb201, target, seed, max(budgets), n_test, tokenizer)
            for name in methods:
                model_jobs = [(setting, n) for model, setting, n in jobs if model == name]
                if not model_jobs:
                    continue
                torch.manual_seed(seed)
                model = (GraphRanker() if name == "gnn" else PythiaRanker(args.model_path)).to(device)
                lr = args.gnn_lr if name == "gnn" else args.pythia_lr
                initial = copy.deepcopy(model.state_dict())
                for setting in SETTINGS:
                    pending = [n for job_setting, n in model_jobs if job_setting == setting]
                    if not pending:
                        continue
                    model.load_state_dict(initial)
                    source_seconds = 0.0
                    if setting == "source_then_target":
                        print(f"{source}->{target} seed={seed} {name}, fitting source", flush=True)
                        start = time.perf_counter()
                        fit(model, source_data.inputs[name], source_data.labels, source_steps,
                            config["batch_size"], lr, seed + 20_000, device)
                        source_seconds = time.perf_counter() - start
                    checkpoint = copy.deepcopy(model.state_dict()) if len(pending) > 1 else None
                    for n in pending:
                        print(f"{source}->{target} seed={seed} {name} {setting} n={n}", flush=True)
                        if checkpoint is not None:
                            model.load_state_dict(checkpoint)
                        start = time.perf_counter()
                        inputs = target_data.inputs[name]
                        fit(model, tuple(t[:n] for t in inputs), target_data.labels[:n],
                            target_steps, config["batch_size"], lr, seed, device)
                        target_seconds = time.perf_counter() - start
                        scores = predict(model, inputs, max(budgets), 32, device)
                        metrics = evaluate(scores, target_data.labels[max(budgets):])
                        rows.append(dict(zip(KEY, (source, target, seed, n, name, setting)),
                                         source_seconds=source_seconds, target_seconds=target_seconds,
                                         **metrics))
                        report(rows, out)
                        print(f"    tau={metrics['tau']:.3f}, regret@10={metrics['regret10']:.3f}", flush=True)
                del model, initial, checkpoint
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
