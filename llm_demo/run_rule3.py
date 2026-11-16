"""规律 3 的总调度：按顺序跑完所有步骤，每一步的输出都追加到 run_rule3.log。

    1. 等 6 个新模型下载完（download_models.py 在另一个进程里下载）
    2. 70M、160M（随机 + 训练完成）在 5 种写法 × 3 个空间上提特征（快，先出结果）
    3. 评估：70M、160M、410M
    4. 1B（随机 + 训练完成）提特征（最慢）
    5. 评估：全部 4 种大小
已经算好的特征和评估结果都会自动跳过，中途停了重新运行这个脚本就能接着跑。

运行方式（在 llm_demo 文件夹下）:
    python run_rule3.py
"""

import subprocess
import sys
import time
from pathlib import Path

from models import MODEL_DIR, has_weights

HERE = Path(__file__).parent
LOG = HERE / "run_rule3.log"
FEATURES = str(HERE / "features_generated_v3" / "formats")
SMALL = ["pythia70m_step0", "pythia70m_step143000", "pythia160m_step0", "pythia160m_step143000"]
BIG = ["pythia1b_step0", "pythia1b_step143000"]
OLD = ["pythia410m_step0", "pythia410m_step143000"]


def say(text):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n========== {time.strftime('%H:%M')}  {text} ==========\n")


def run(*args):
    say("运行 " + " ".join(args))
    with open(LOG, "a", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, "-u", *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    if code != 0:
        say(f"出错了（返回码 {code}），停在这里")
        raise SystemExit(code)


def wait_for(models, timeout=21600):
    say("等模型下载完：" + ", ".join(models))
    deadline = time.monotonic() + timeout
    while not all((MODEL_DIR / m / ".download_done").exists() and has_weights(MODEL_DIR / m) for m in models):
        if time.monotonic() >= deadline:
            raise TimeoutError("Model downloads not ready within timeout")
        time.sleep(30)


def main():
    wait_for(SMALL)
    run("run_formats.py", "--models", ",".join(SMALL + OLD), "--out", FEATURES)
    run("step3_formats.py", "--models", ",".join(SMALL + OLD), "--feat-dir", FEATURES)
    wait_for(BIG)  # 1B 比较大，下载和前面的步骤同时进行
    run("run_formats.py", "--models", ",".join(BIG), "--out", FEATURES)
    run("step3_formats.py", "--models", "all", "--feat-dir", FEATURES)
    say("规律 3 全部完成 EXIT")


if __name__ == "__main__":
    main()
