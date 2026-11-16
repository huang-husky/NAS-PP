"""用 DeepSeek（强模型）重做"留一操作"检验：问题是"之前失败，是不是因为开源小模型太弱？"

为什么不再用"嵌入"：DeepSeek 的官方接口是聊天接口，拿不到隐藏状态，没法像 Qwen 那样取向量。
所以换一种更直接、对强模型更公平的问法——让它直接当预测器：

    k=0   不给任何带标签的例子，只告诉它 5 种操作是什么，请它预测准确率（纯先验知识）。
    k=30  给 30 个"不含被留出操作"的带标签架构当例子（in-context），再请它预测含被留出操作的架构。

对照（用同样的 30 个标签）：one-hot 随机森林、手写属性随机森林。另外给一个 n=500 的参考。
评估：对每个被留出的操作，随机抽 100 个含它的架构，算预测与真实准确率的 Spearman。

放弃条件（跑之前写下）
    DeepSeek(k=30) 的平均 Spearman，比"同样 30 个标签的 one-hot 随机森林"和
    "同样 30 个标签的手写属性随机森林"里较好的那个，高不到 0.05
        -> 说明"开源小模型太弱"不是主要原因，这个方向停止。

API key：只从环境变量 DEEPSEEK_API_KEY 读，不写进任何文件，不打印。

用法
    python run_deepseek.py --dry-run     不联网：看提示词长什么样、估算花费、先算好对照结果
    python run_deepseek.py --smoke       联网，只发 2 次请求，确认 key 能用
    python run_deepseek.py               全部跑（约 100 次请求，几分钟，花费通常不到几毛钱）
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))      # python -I 不会自动把脚本所在文件夹加入搜索路径
from run_op_holdout import (DATA, DESC_NAT, EDGES, HAND, OPS, OUT, TARGET,  # noqa: E402
                            arch_features, fit_predict)

API_URL = "https://api.deepseek.com/chat/completions"
N_TEST = 100        # 每个被留出的操作，评估多少个含它的架构
BATCH = 20          # 每次请求让模型预测几个架构
K_SHOT = 30         # k=30 时给几个带标签的例子
SEEDS = [0, 1, 2]   # k=30 时用不同的例子集合重复几次
CACHE = OUT / "deepseek_cache.jsonl"


# ======================================================================================
# 第 1 段：把一个架构写成 NAS-Bench-201 的标准字符串
# ======================================================================================
def arch_string(row):
    """|op_0to1~0|+|op_0to2~0|op_1to2~1|+|op_0to3~0|op_1to3~1|op_2to3~2|"""
    o = {e: row[e] for e in EDGES}
    return (f"|{o['op_0to1']}~0|+|{o['op_0to2']}~0|{o['op_1to2']}~1|+"
            f"|{o['op_0to3']}~0|{o['op_1to3']}~1|{o['op_2to3']}~2|")


# ======================================================================================
# 第 2 段：提示词
# ======================================================================================
HEADER = (
    "NAS-Bench-201 describes a small convolutional network by a cell: a DAG with 4 nodes "
    "(node 0 is the input, node 3 the output). Each of the 6 edges "
    "(0->1, 0->2, 1->2, 0->3, 1->3, 2->3) carries exactly one operation, and a node sums the "
    "outputs of its incoming edges. Architecture string format:\n"
    "|op(0->1)~0|+|op(0->2)~0|op(1->2)~1|+|op(0->3)~0|op(1->3)~1|op(2->3)~2|\n\n"
    "The five possible operations are:\n"
    + "\n".join(f"- {o}: {DESC_NAT[o]}" for o in OPS)
    + "\n\nThe cell is stacked into a full network, trained with the standard NAS-Bench-201 protocol, "
    "and its CIFAR-10 test accuracy (in %) is recorded."
)


def build_prompt(examples, queries):
    """examples: [(架构字符串, 准确率)]，可为空；queries: [架构字符串]"""
    p = HEADER + "\n\n"
    if examples:
        p += f"Here are {len(examples)} architectures with their measured accuracy:\n"
        p += "\n".join(f"{a} -> {acc:.2f}" for a, acc in examples) + "\n\n"
    p += (f"Predict the CIFAR-10 test accuracy (%) of each of the following {len(queries)} architectures. "
          f"Reply with ONLY a JSON list of {len(queries)} numbers, in the same order, and nothing else.\n")
    p += "\n".join(f"{i + 1}. {a}" for i, a in enumerate(queries))
    return p


# ======================================================================================
# 第 3 段：调用 API（带缓存：同一个提示词只付一次钱，中断后重跑会接着来）
# ======================================================================================
def load_cache():
    d = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            d[r["key"]] = r["reply"]
    return d


THINK = None    # DeepSeek：None=不设置（和第一次运行完全一样）；"on"=开启思考；"off"=关闭思考
PROVIDER = "deepseek"   # "deepseek" 或 "kimi"
EFFORT = None           # Kimi K3 系列：推理强度 low / high / max（K3 不能关闭思考）
USAGE = {"calls": 0, "prompt": 0, "completion": 0}   # 本次运行实际用掉的词元，用来核对花费


def call_api(prompt, model, key, cache):
    import requests
    # 缓存键包含模型和设置：换模型或换设置不会误用旧回复。
    # DeepSeek 的键格式保持不变，所以之前的缓存仍然有效。
    extra = (f"|thinking={THINK}" if THINK else "") + (f"|kimi|effort={EFFORT}" if PROVIDER == "kimi" else "")
    if PROVIDER == "kimi" and model.startswith("kimi-k2"):
        extra += "|mct=4000"      # 长度上限变了：让之前被截断的缓存回复失效
    k = hashlib.sha256((model + extra + "\n" + prompt).encode()).hexdigest()
    if k in cache:
        return cache[k]
    k2 = PROVIDER == "kimi" and model.startswith("kimi-k2")
    if k2:
        # Kimi K2.x：可以关闭思考（K3 不行），回复很快。文档没说明 temperature 的限制，所以不设置。
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "thinking": {"type": "disabled"}, "max_completion_tokens": 4000}
    elif PROVIDER == "kimi":
        # Kimi K3：始终思考；用 reasoning_effort 控制强度；长度用 max_completion_tokens（思考也算在内）。
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "reasoning_effort": EFFORT, "max_completion_tokens": 32000}
    else:
        payload = {"model": model, "temperature": 0, "max_tokens": 500,
                   "messages": [{"role": "user", "content": prompt}]}
        if THINK == "off":
            payload["thinking"] = {"type": "disabled"}
        elif THINK == "on":
            # 思考模式：不支持 temperature（设了也无效）；思考过程也占输出长度，所以放宽上限
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = "high"
            payload["max_tokens"] = 16000
    slow = (PROVIDER == "kimi" and not k2) or THINK == "on"
    for attempt in range(5):
        try:
            r = requests.post(API_URL, timeout=900 if slow else 180,
                              headers={"Authorization": f"Bearer {key}"}, json=payload)
            if r.status_code == 200:
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                u = data.get("usage", {})
                USAGE["calls"] += 1
                USAGE["prompt"] += u.get("prompt_tokens", 0)
                USAGE["completion"] += u.get("completion_tokens", 0)
                if data["choices"][0].get("finish_reason") == "length":
                    # 被截断的回复不写进缓存，否则重跑还会读到残缺的回复
                    print("  警告：回复因长度上限被截断，这批可能解析失败", flush=True)
                    return reply
                with open(CACHE, "a", encoding="utf-8") as f:     # 追加写入，不含 key
                    f.write(json.dumps({"key": k, "reply": reply}, ensure_ascii=False) + "\n")
                cache[k] = reply
                return reply
            print(f"  请求失败 HTTP {r.status_code}（第 {attempt + 1} 次）：{r.text[:150]}", flush=True)
            if r.status_code == 401:       # key 不对或地址不对：重试没有意义
                return None
            if r.status_code == 429:       # 请求太频繁：多等一会儿
                time.sleep(20 * (attempt + 1))
        except Exception as e:                                      # 网络抖动
            print(f"  网络错误（第 {attempt + 1} 次）：{type(e).__name__}", flush=True)
        time.sleep(3 * (attempt + 1))
    return None


def parse_numbers(reply, n):
    """从回复里取出 n 个数字；取不到返回 None。"""
    if reply is None:
        return None
    # 有的模型（如 Kimi K2.6）会先写一大段分析，最后才给列表：所以从后往前找，
    # 取最后一个"恰好有 n 个数字"的方括号列表，不会被分析文字里的数字干扰。
    for m in reversed(re.findall(r"\[[^\[\]]*\]", reply)):
        try:
            arr = [float(x) for x in json.loads(m)]
        except Exception:
            continue
        if len(arr) == n:
            return arr
    if len(reply) < 400:       # 很短的回复才允许退而求其次：直接抓数字（例如没写方括号）
        arr = [float(x) for x in re.findall(r"-?\d+\.?\d*", reply)]
        if len(arr) == n:
            return arr
    return None


# ======================================================================================
# 第 4 段：主流程
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seen-control", action="store_true",
                    help="对照：例子里允许出现被留出的操作（操作'见过'），看优势是不是只在没见过时才有")
    ap.add_argument("--model", default="deepseek-chat",
                    help="模型名，如 deepseek-v4-pro（最高档）、deepseek-flash；默认是第一次用的 deepseek-chat")
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="思考模式：on 开启（更慢更贵），off 关闭；不写则不设置")
    ap.add_argument("--provider", choices=["deepseek", "kimi"], default="deepseek")
    ap.add_argument("--effort", choices=["low", "high", "max"], default="low",
                    help="仅 Kimi K3：推理强度（越高越慢越贵）。默认 low，和之前'不怎么思考'的设置最接近")
    ap.add_argument("--base-url", default=None,
                    help="完整的聊天接口地址。Kimi 默认国内站 https://api.moonshot.cn/v1/chat/completions；"
                         "如果 key 来自国际站，改成 https://api.moonshot.ai/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=None, help="同时发几个请求（Kimi 默认 2，DeepSeek 默认 4）")
    args = ap.parse_args()
    global THINK, API_URL, EFFORT, PROVIDER
    THINK, PROVIDER = args.thinking, args.provider
    if PROVIDER == "kimi":
        is_k2 = args.model.startswith("kimi-k2")
        EFFORT = None if is_k2 else args.effort
        if is_k2:
            THINK = "off"          # K2.x 支持关闭思考；缓存键也因此区别于 K3
        # 默认国内站（platform.kimi.com，接口 api.moonshot.cn）。国际站的 key 与国内站不能混用。
        API_URL = args.base_url or "https://api.moonshot.cn/v1/chat/completions"
        # 环境变量名：官方示例用 MOONSHOT_API_KEY，也接受 KIMI_API_KEY
        key_env = "MOONSHOT_API_KEY" if os.environ.get("MOONSHOT_API_KEY") else "KIMI_API_KEY"
        tag = f"{args.model}-think-off" if is_k2 else f"{args.model}-effort-{args.effort}"
        args.workers = args.workers or (4 if is_k2 else 6)
        print(f"本次使用：Kimi，模型 {args.model}，"
              f"{'思考已关闭' if is_k2 else '推理强度 ' + str(EFFORT)}，并发 {args.workers}，地址 {API_URL}", flush=True)
    else:
        API_URL = args.base_url or API_URL
        key_env = "DEEPSEEK_API_KEY"
        # 结果文件名带上模型和思考设置，不同模型的结果互不覆盖
        tag = args.model + (f"-think-{args.thinking}" if args.thinking else "")
        args.workers = args.workers or 4
        print(f"本次使用模型：{args.model}，思考模式：{args.thinking or '未设置（接口默认）'}", flush=True)

    key = os.environ.get(key_env, "")
    if not args.dry_run and not key:
        sys.exit(f"没有找到环境变量 {key_env}。设置方法见说明；key 不要写进代码或发到聊天里。")

    df = pd.read_csv(DATA).dropna(subset=[TARGET]).reset_index(drop=True)
    op_idx = df[EDGES].apply(lambda s: s.map(OPS.index)).values
    y = df[TARGET].values
    strs = [arch_string(r) for _, r in df.iterrows()]
    Xone = arch_features(op_idx, np.eye(5))
    Xhand = arch_features(op_idx, np.array([HAND[o] for o in OPS], dtype=float))

    held_ops = OPS[3:4] if args.smoke else OPS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    jobs, base_rows = [], []            # jobs: 要发的请求；base_rows: 对照结果
    for held in held_ops:
        h = OPS.index(held)
        has_h = (op_idx == h).any(1)
        pool = np.where(~has_h)[0]
        test = np.random.default_rng(100 + h).choice(np.where(has_h)[0], N_TEST, replace=False)
        batches = [test[i:i + BATCH] for i in range(0, N_TEST, BATCH)]
        if args.smoke:
            batches = batches[:1]
        # k=0：没有例子
        for bi, b in enumerate(batches):
            jobs.append(dict(held=held, variant="ds_k0", seed=0, bi=bi, test=b, ex=[]))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ex_idx = rng.choice(pool, K_SHOT, replace=False)
            ex = [(strs[i], y[i]) for i in ex_idx]
            for bi, b in enumerate(batches):
                jobs.append(dict(held=held, variant="ds_k30", seed=seed, bi=bi, test=b, ex=ex))
            # 同样 30 个标签的对照，以及 n=500 的参考（全部在同一批 100 个测试架构上）
            tr500 = rng.choice(pool, 500, replace=False)
            for name, X, tr in [("rf_onehot_k30", Xone, ex_idx), ("rf_hand_k30", Xhand, ex_idx),
                                ("rf_onehot_n500", Xone, tr500)]:
                p = fit_predict("rf", X[tr], y[tr], X[test], seed)
                base_rows.append(dict(held=held, variant=name, seed=seed,
                                      rho=spearmanr(p, y[test])[0]))
            if args.seen_control:
                # 对照：30 个例子从"所有架构"（除测试集本身）里抽，因此包含被留出的操作
                pool_all = np.setdiff1d(np.arange(len(df)), test)
                ex_seen = np.random.default_rng(seed + 50).choice(pool_all, K_SHOT, replace=False)
                exs = [(strs[i], y[i]) for i in ex_seen]
                for bi, b in enumerate(batches):
                    jobs.append(dict(held=held, variant="ds_k30_seen", seed=seed, bi=bi, test=b, ex=exs))
                for name, X in [("rf_onehot_k30_seen", Xone), ("rf_hand_k30_seen", Xhand)]:
                    p = fit_predict("rf", X[ex_seen], y[ex_seen], X[test], seed)
                    base_rows.append(dict(held=held, variant=name, seed=seed,
                                          rho=spearmanr(p, y[test])[0]))
        if args.smoke:
            break

    # ---- 干跑：不联网 ----
    if args.dry_run:
        ex_job = next(j for j in jobs if j["variant"] == "ds_k30")
        sample = build_prompt(ex_job["ex"], [strs[i] for i in ex_job["test"]])
        tot_chars = sum(len(build_prompt(j["ex"], [strs[i] for i in j["test"]])) for j in jobs)
        print("=" * 30, "提示词示例（k=30）", "=" * 30)
        print(sample[:1800], "\n……（中间省略）……\n", sample[-500:])
        print("=" * 80)
        print(f"请求数：{len(jobs)}；提示词总长约 {tot_chars} 字符 ≈ {tot_chars / 3.5 / 1e3:.0f}K 词元；")
        if PROVIDER == "kimi":
            print("Kimi 的价格我没能从官方页面查到（价格表是空的），所以不估算花费。"
                  "K3 始终思考，输出词元可能比输入多得多：请先跑 --smoke，看它打印的实际用量再决定。")
        else:
            print(f"粗估花费：约 ${tot_chars / 3.5 / 1e6 * 0.14 + len(jobs) * 150 / 1e6 * 0.28:.3f}"
                  f"（按每百万输入 $0.14、输出 $0.28 估算，实际以官网价格为准）")
        b = pd.DataFrame(base_rows).groupby(["variant", "held"]).rho.mean().unstack("held")
        b["平均"] = b.mean(1)
        print("\n对照结果（同样的测试集，无需联网）：\n", b.round(3).to_string())
        return

    # ---- 真跑 ----
    cache = load_cache()
    done, t0 = [0], time.time()

    def run(job):
        prompt = build_prompt(job["ex"], [strs[i] for i in job["test"]])
        nums = parse_numbers(call_api(prompt, args.model, key, cache), len(job["test"]))
        if nums is None:        # 格式不对时再问一次（提示词一样但缓存里没有成功的回复）
            nums = parse_numbers(call_api(prompt + "\n(Reply with the JSON list only.)",
                                          args.model, key, cache), len(job["test"]))
        done[0] += 1
        el = time.time() - t0
        bar = "#" * int(20 * done[0] / len(jobs)) + "-" * (20 - int(20 * done[0] / len(jobs)))
        print(f"[{bar}] {done[0]}/{len(jobs)}  已用 {el / 60:.1f} 分钟，预计剩余 "
              f"{el / done[0] * (len(jobs) - done[0]) / 60:.1f} 分钟", flush=True)
        return job, nums

    with ThreadPoolExecutor(args.workers) as pool_:
        results = list(pool_.map(run, jobs))
    print(f"\n本次实际用量：{USAGE['calls']} 次请求，输入 {USAGE['prompt'] / 1e3:.0f}K 词元，"
          f"输出 {USAGE['completion'] / 1e3:.0f}K 词元（输出包含思考过程；对照控制台账单核对花费）", flush=True)

    # 把同一 (操作, 变体, 种子) 的各批预测拼回 100 个，再算 Spearman
    rows = list(base_rows)
    groups = {}
    for job, nums in results:
        g = groups.setdefault((job["held"], job["variant"], job["seed"]), [])
        g.append((job["bi"], job["test"], nums))
    for (held, variant, seed), g in groups.items():
        g.sort(key=lambda t: t[0])
        idx = np.concatenate([t[1] for t in g])
        pred = np.concatenate([t[2] if t[2] is not None else [np.nan] * len(t[1]) for t in g])
        ok = ~np.isnan(pred)
        rho = spearmanr(pred[ok], y[idx][ok])[0] if ok.sum() > 10 else np.nan
        rows.append(dict(held=held, variant=variant, seed=seed, rho=rho, n_valid=int(ok.sum())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"deepseek_{tag}_raw{'_smoke' if args.smoke else ''}.csv", index=False)

    tab = res.groupby(["variant", "held"]).rho.mean().unstack("held")
    tab["平均"] = tab.mean(1)
    order = ["rf_onehot_k30", "rf_hand_k30", "rf_onehot_n500", "ds_k0", "ds_k30",
             "rf_onehot_k30_seen", "rf_hand_k30_seen", "ds_k30_seen"]
    print("\n===== 被留出操作后的 Spearman（越高越好）=====")
    print(tab.loc[[v for v in order if v in tab.index]].round(3).to_string())
    tab.round(4).to_csv(OUT / f"deepseek_{tag}_summary{'_smoke' if args.smoke else ''}.csv")
    if "ds_k30" in tab.index and not args.smoke:
        ds = tab.loc["ds_k30", "平均"]
        best = max(tab.loc["rf_onehot_k30", "平均"], tab.loc["rf_hand_k30", "平均"])
        print(f"\n放弃条件：DeepSeek(k=30) {ds:.3f} - 同标签数对照中较好者 {best:.3f} = {ds - best:+.3f}"
              f"（要求 >= +0.05）-> {'通过' if ds - best >= 0.05 else '不通过'}")
    if "ds_k30_seen" in tab.index and not args.smoke:
        # 第二条标准（在看到 DeepSeek 对照结果之前修订：只有 30 个标签时随机森林噪声太大，
        # 它"见过 vs 没见过"的差没有意义，所以只看 DeepSeek 自己）：
        #   g = DeepSeek 见过操作 - DeepSeek 没见过操作
        #   g >= 0.05：强模型面对没见过的操作仍然明显变差，"新操作"问题依旧存在；
        #   g <  0.05：强模型基本不受"没见过"影响，之前的失败更可能是小模型能力不足。
        g_ds = tab.loc["ds_k30_seen", "平均"] - tab.loc["ds_k30", "平均"]
        print(f"\nDeepSeek 见过操作 {tab.loc['ds_k30_seen', '平均']:.3f}，没见过 {tab.loc['ds_k30', '平均']:.3f}，"
              f"差 g = {g_ds:+.3f}（>= 0.05 说明新操作问题仍在；< 0.05 说明强模型基本不受影响）")
        print("（随机森林 30 个标签的'见过/没见过'仅作参考，噪声很大："
              f"{tab.loc['rf_onehot_k30_seen', '平均']:.3f} / {tab.loc['rf_onehot_k30', '平均']:.3f}）")


if __name__ == "__main__":
    main()
