"""观察微调扫描的进度（只读，不影响正在跑的任务）。

显示：每个学习率完成了几次、已有结果的平均 τ、正在跑的是哪一个、预计还要多久。
    python watch_progress.py              看一次
    python watch_progress.py --watch      每 30 秒刷新一次（按 Ctrl+C 退出）

默认对应当前的学习率扫描；以后换了实验，用参数改：
    python watch_progress.py --tag dev_A0 --lrs 5e-5,2e-4,5e-4 --seeds 100,101,102 --n 200
"""

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OVERHEAD_MIN = 0.6          # 每次除了训练之外，加载模型和预测测试集大约多花的分钟数（按前几次的实测估的）


def read_rows(tag):
    f = RESULTS / f"ft_{tag}.csv"
    if not f.exists():
        return [], None
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, datetime.fromtimestamp(f.stat().st_mtime)


def running_job():
    """在系统里找正在跑的 finetune.py，返回它的命令行（找不到返回 None）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*finetune.py*' } | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return out or None
    except Exception:
        return None


def show(args):
    lrs = [float(x) for x in args.lrs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows, last_write = read_rows(args.tag)
    rows = [r for r in rows if int(r["n"]) == args.n]
    total = len(lrs) * len(seeds)
    done = 0
    print(f"=== 微调扫描进度（{args.tag}，n={args.n}）   现在 {datetime.now():%H:%M:%S} ===\n")
    for lr in lrs:
        got = {int(r["seed"]): float(r["tau"]) for r in rows if abs(float(r["lr"]) - lr) < 1e-12}
        done += len(got)
        bar = "#" * len(got) + "-" * (len(seeds) - len(got))
        mean = f"平均 τ = {sum(got.values()) / len(got):.3f}" if got else "还没有结果"
        detail = "  ".join(f"种子{s}:{got[s]:.3f}" if s in got else f"种子{s}:--" for s in seeds)
        print(f"  lr={lr:<8g} [{bar}] {len(got)}/{len(seeds)}   {mean}")
        print(f"             {detail}")

    cmd = running_job()
    print()
    if cmd:
        m_lr = re.search(r"--lr\s+(\S+)", cmd)
        m_seeds = re.search(r"--seeds\s+(\S+)", cmd)
        print(f"  正在跑：lr={m_lr.group(1) if m_lr else '?'}，种子 {m_seeds.group(1) if m_seeds else '?'}")
    else:
        print("  现在没有微调进程在运行" + ("（任务已结束或中断）" if done < total else "（全部完成）"))

    remaining = total - done
    if rows and remaining > 0:
        per_run = sum(float(r["minutes"]) for r in rows) / len(rows) + OVERHEAD_MIN
        base = last_write if cmd else datetime.now()
        eta = base + timedelta(minutes=remaining * per_run)
        left = max(0.0, (eta - datetime.now()).total_seconds() / 60)
        print(f"  总进度 {done}/{total}，每次约 {per_run:.1f} 分钟，还剩 {remaining} 次，"
              f"预计还要 {left:.0f} 分钟（约 {eta:%H:%M} 完成）")
        if not cmd:
            print("  （没有进程在跑，所以没有预计完成时间）")
    elif remaining == 0:
        print(f"  总进度 {done}/{total}，全部完成")
    else:
        print(f"  总进度 {done}/{total}，还没有可用来估计耗时的结果")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev_A0")
    ap.add_argument("--lrs", default="5e-5,2e-4,5e-4")
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    if not args.watch:
        show(args)
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            show(args)
            print("\n（每 30 秒刷新一次，Ctrl+C 退出）")
            time.sleep(30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
