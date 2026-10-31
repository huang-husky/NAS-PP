#!/usr/bin/env bash
# 晚上挂机：等第 1 步（explain.py）跑完，再依次提取第 2 步和 TransNAS 的特征。
# 日志写到 run_overnight.log；每个模型每 2 分钟追加一行进度。
# 已经算好的文件会自动跳过，所以中途断了重新运行这个脚本就能接着跑。
cd "$(dirname "$0")"
LOG=run_overnight.log
P2="pythia410m_step0,pythia410m_step143000"

echo "$(date +%H:%M) 等第 1 步跑完……" >> $LOG
until grep -q "EXIT" run_step1.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M) 第 1 步结束，开始提取特征" >> $LOG

run() { echo; echo "========== $(date +%H:%M)  $* =========="; python -u embed.py "$@"; } >> $LOG 2>&1

# 1. TransNAS：3 个代表性模型，4096 个结构，最快出结果（约 40 分钟）
run --space transnas --only pythia410m_step0,pythia410m_step143000,qwen25_0.5b
# 2. 2×2 实验的三种新情况，两个 Pythia（每个约 50 分钟）
run --shuffle-ops --only $P2
run --anonymous --only $P2
run --anonymous --shuffle-ops --only $P2
# 3. 原始文本的分行特征（平均特征已有，不会覆盖）
run --only $P2

echo "$(date +%H:%M) 全部完成 EXIT" >> $LOG
