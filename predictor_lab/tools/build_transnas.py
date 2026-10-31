"""从 TransNAS-Bench-101 的 MindSpore 版数据生成 data/transnas_micro.csv。

原始数据: https://download.mindspore.cn/dataset/TransNAS-Bench-101/
需要其中两个文件: micro.mindrecord（约 219MB）和 micro.mindrecord.db（索引，约 270KB）。

MindRecord 文件的格式（不用装 MindSpore，自己读）:
    开头 8 字节是一个整数 = 文件头 JSON 的长度，后面跟着这段 JSON，
    JSON 里的 header_size 和 page_size 告诉我们：第 k 页从 header_size + k * page_size 处开始。
    .db 是一个 SQLite 索引，每个结构一行，写着它的数据在第几页、页内从哪到哪。
    每个结构的数据 = 8 字节长度 + 一段 JSON 文本（只是文本，读取时不会执行任何代码）。

micro（单元级）搜索空间:
    和 NAS-Bench-201 几乎一样：4 个节点、6 条边，每条边选一种操作，
    只是操作只有 4 种（没有池化）: 0 = none, 1 = skip_connect, 2 = nor_conv_1x1, 3 = nor_conv_3x3。
    结构字符串形如 "64-41414-0_00_123"，最后一段 "0_00_123" 依次是 6 条边的操作:
        节点1 <- 节点0；节点2 <- 节点0, 1；节点3 <- 节点0, 1, 2（边的顺序和 NB201 的 EDGES 一致）
    一共 4^6 = 4096 个结构，每个结构在 7 个任务上各训练过一次。

每个任务取论文里用的主要指标，在最后一轮的验证集结果（数值越大越好）:
    class_scene / class_object / jigsaw  -> top1 准确率
    room_layout                          -> neg_loss（负的损失）
    segmentsemantic                      -> mIoU
    normal / autoencoder                 -> ssim

用法:
    python tools/build_transnas.py <存放 micro.mindrecord 和 micro.mindrecord.db 的文件夹>
"""

import csv
import json
import sqlite3
import struct
import sys
from pathlib import Path

OUT_FILE = Path(__file__).parent.parent / "data" / "transnas_micro.csv"
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3"]
MAIN_METRIC = {
    "class_scene": "top1", "class_object": "top1", "jigsaw": "top1", "room_layout": "neg_loss",
    "segmentsemantic": "mIoU", "normal": "ssim", "autoencoder": "ssim",
}


def read_records(src_dir):
    """依次产出 (结构字符串, 这个结构的全部数据)。"""
    raw = (src_dir / "micro.mindrecord").read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len].rstrip(b"\x00"))
    base, page = header["header_size"], header["page_size"]

    db = sqlite3.connect(f"file:{src_dir / 'micro.mindrecord.db'}?mode=ro", uri=True)
    cols = [c[1] for c in db.execute("PRAGMA table_info(INDEXES)")]
    i_page, i_start, i_end = cols.index("PAGE_ID_BLOB"), cols.index("PAGE_OFFSET_BLOB"), cols.index("PAGE_OFFSET_BLOB_END")
    for row in db.execute("SELECT * FROM INDEXES ORDER BY ROW_ID"):
        start = base + row[i_page] * page
        blob = raw[start + row[i_start]: start + row[i_end]]
        n = struct.unpack("<Q", blob[:8])[0]
        yield row[-1], json.loads(blob[8:8 + n])  # 最后一列是 label，也就是结构字符串


def main(src_dir):
    src_dir = Path(src_dir)
    tasks = list(MAIN_METRIC)
    header = ["arch_str"] + [f"op_{e}" for e in ["0to1", "0to2", "1to2", "0to3", "1to3", "2to3"]]
    header += [f"{t}_valid" for t in tasks] + [f"{t}_test" for t in tasks]
    header += ["encoder_params", "encoder_FLOPs"]

    OUT_FILE.parent.mkdir(exist_ok=True)
    n_rows, unfinished = 0, 0
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for arch_str, data in read_records(src_dir):
            codes = arch_str.split("-")[-1].replace("_", "")  # "0_00_123" -> "000123"
            assert len(codes) == 6, arch_str
            valid, test = [], []
            for t in tasks:
                m = data[t]["metrics"]
                unfinished += not data[t]["finish_train"]
                valid.append(m[f"valid_{MAIN_METRIC[t]}"][-1])
                test.append(m[f"test_{MAIN_METRIC[t]}"][-1])
            info = data["class_scene"]["model_info"]  # 编码器（也就是 cell 堆起来的主干）在各任务都一样
            writer.writerow([arch_str] + [OPS[int(c)] for c in codes] + valid + test
                            + [info["encoder_params"], info["encoder_FLOPs"]])
            n_rows += 1

    print(f"共 {n_rows} 个结构（应为 4096），没训练完的 任务×结构 数: {unfinished}")
    print(f"已写入 {OUT_FILE}")


if __name__ == "__main__":
    main(sys.argv[1])
