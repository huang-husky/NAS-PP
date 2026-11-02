"""COLE 写法检验，按顺序自动跑完，输出追加到 run_cole.log:
    1. 用 COLE 的写法提取随机 / 训练完成 Pythia-410M 的特征（NB201 5000 个，TransNAS 4096 个）
    2. step7_cole.py：词频上限 + 学习曲线（岭回归、随机森林、MLP）

运行方式（在 llm_demo 文件夹下）:
    python run_cole.py
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "run_cole.log"


def run(*args):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n========== {time.strftime('%H:%M')}  运行 {' '.join(args)} ==========\n")
        f.flush()
        code = subprocess.run([sys.executable, "-u", *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        if code != 0:
            f.write(f"\n========== {time.strftime('%H:%M')}  出错了（返回码 {code}），停在这里 ==========\n")
            raise SystemExit(code)


if __name__ == "__main__":
    run("run_formats.py", "--spaces", "nb201,transnas", "--formats", "cole")
    run("step7_cole.py")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n========== {time.strftime('%H:%M')}  COLE 检验全部完成 EXIT ==========\n")
