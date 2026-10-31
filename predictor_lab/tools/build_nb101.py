"""从 NAS-Bench-101 的 SQLite 数据库生成 data/nasbench101.csv。

原始数据: https://huggingface.co/datasets/cShahar/nas-benchmarks （nasbench101-209f5694.db，约 2.1GB）
这是 NNI 工具包整理的格式，里面有三张表:
    nb101trialconfig   每个结构 × 每种训练轮数（4 / 12 / 36 / 108）一行，arch 列是结构的 JSON
    nb101trialstats    每次训练一行（每个结构、每种轮数各训练 3 次），有精度和参数量
    nb101intermediatestats  训练过程中的中间结果（这里用不到）

NAS-Bench-101 的结构:
    一个 cell 最多 7 个节点：节点 0 是输入，最后一个是输出，中间最多 5 个节点，
    每个中间节点选一种操作：conv3x3-bn-relu / conv1x1-bn-relu / maxpool3x3。
    节点之间怎么连可以随意（只要是从前往后连、总边数不超过 9），所以比 NB201 灵活得多。
    arch JSON 里 "op1"～"op5" 是中间节点的操作，"input1"～"input6" 是每个节点的输入来自哪些节点。

用法:
    python tools/build_nb101.py <nasbench101-209f5694.db 的路径>
"""

import csv
import sqlite3
import sys
from pathlib import Path

OUT_FILE = Path(__file__).parent.parent / "data" / "nasbench101.csv"

# 只取训练 108 轮的结果（论文里的主结果），3 次训练取平均
QUERY = """
SELECT c.hash, c.arch, c.num_vertices,
       AVG(s.test_acc), AVG(s.valid_acc), AVG(s.train_acc), AVG(s.parameters), COUNT(*)
FROM nb101trialconfig AS c JOIN nb101trialstats AS s ON s.config_id = c.id
WHERE c.num_epochs = 108
GROUP BY c.id
"""


def main(db_path):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # 只读打开，不会改动原文件
    rows = db.execute(QUERY).fetchall()

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "arch", "num_vertices", "test_acc", "valid_acc", "train_acc",
                         "params_M", "n_trials"])
        for h, arch, nv, test, valid, train, params, n in rows:
            writer.writerow([h, arch, nv, round(test, 4), round(valid, 4), round(train, 4), params, n])

    # 简单自查：结构数应为 423624，最好的平均测试精度应约为 94.32%（论文数字）
    tests = [r[3] for r in rows]
    print(f"共 {len(rows)} 个结构，测试精度 平均 {sum(tests) / len(tests):.2f}% ，最高 {max(tests):.2f}%")
    print(f"每个结构训练次数: {sorted({r[7] for r in rows})}")
    print(f"已写入 {OUT_FILE}")


if __name__ == "__main__":
    main(sys.argv[1])
