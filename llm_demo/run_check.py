"""汇报前的补充实验（"体检"），按顺序自动跑完，输出追加到 run_check.log:
    1. Qwen2.5-0.5B 也做一遍 5 种写法的实验（检验规律 2 是不是只在 Pythia 上成立）
    2. 把 Qwen 加进写法实验的评估（其他模型的结果已经算过，会直接复用）
    3. 搜索实验从 20 次加到 50 次，并加上 Qwen（已经跑过的 20 次会直接复用）

运行方式（在 llm_demo 文件夹下）:
    python run_check.py
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "run_check.log"
PYTHIA = [f"pythia{s}_{v}" for s in ["70m", "160m", "410m", "1b"] for v in ["step0", "step143000"]]


def run(*args):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n========== {time.strftime('%H:%M')}  运行 {' '.join(args)} ==========\n")
        f.flush()
        code = subprocess.run([sys.executable, "-u", *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        if code != 0:
            f.write(f"\n========== {time.strftime('%H:%M')}  出错了（返回码 {code}），停在这里 ==========\n")
            raise SystemExit(code)


if __name__ == "__main__":
    run("run_formats.py", "--models", "qwen25_0.5b")
    run("step3_formats.py", "--models", ",".join(PYTHIA + ["qwen25_0.5b"]))
    run("step6_search.py", "--runs", "50")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n========== {time.strftime('%H:%M')}  补充实验全部完成 EXIT ==========\n")
