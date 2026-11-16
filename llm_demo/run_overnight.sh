#!/usr/bin/env bash
set -euo pipefail
# 晚上挂机：先跑第 1 步（explain.py），再提取第 2 步和 TransNAS 的新特征，最后用新特征跑 step2。
# 日志写到 run_overnight.log；每个模型每 2 分钟追加一行进度。
# 已经算好的文件会自动跳过，所以中途断了重新运行这个脚本就能接着跑。
cd "$(dirname "$0")"
LOG=run_overnight.log
P2="pythia410m_step0,pythia410m_step143000"

# Execute the prerequisite and require its actual exit status; never trust an old EXIT log.
python -u explain.py >> "$LOG" 2>&1

run() { echo; echo "========== $(date +%H:%M)  $* =========="; python -u "$@"; } >> $LOG 2>&1

# 新特征全部放进 features_generated_v3/，目录结构和 features/ 一样（NB201 在根目录，TransNAS 在 transnas/），
# 这样最后的 step2.py 用 --feature-dir 就能直接读到。新目录里没有旧特征，所有文件都会重新计算。
NEW=features_generated_v3

# 1. TransNAS：3 个代表性模型，4096 个结构（平均特征 + 分行特征）
run embed.py --space transnas --out-dir $NEW/transnas --only pythia410m_step0,pythia410m_step143000,qwen25_0.5b
# 2. NB201 原始文本：两个 Pythia 的平均特征 + 分行特征
run embed.py --out-dir $NEW --only $P2
# 3. 2×2 实验的三种变体，两个 Pythia
run embed.py --out-dir $NEW --shuffle-ops --only $P2
run embed.py --out-dir $NEW --anonymous --only $P2
run embed.py --out-dir $NEW --anonymous --shuffle-ops --only $P2
# 4. 词频特征，和上面用同一批结构
run make_bow.py --out-root $NEW
# 5. 用这批新特征跑 step2（NB201 的 2×2 / 分行实验 + TransNAS）
run step2.py --feature-dir $NEW

echo "$(date +%H:%M) 全部完成 EXIT" >> $LOG
