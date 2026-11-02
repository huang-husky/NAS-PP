"""生成给孙老师的阶段性进展汇报（Word）。

所有数字都抄自 ../RESULTS.md（各节已注明出处），图来自 ../results/。
运行方式（在 llm_demo 文件夹下）:
    python report/make_report.py
输出: report/进展汇报_2026-11-02.docx，report/fig_scatter_compact.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "进展汇报_2026-11-02.docx"


# ---------------------------------------------------------------- 图 2：精简版散点图（只放 2 个模型）

def compact_scatter(path):
    fmt_color = {"code": "tab:blue", "edges": "tab:orange", "sorted": "tab:green", "nl": "tab:purple",
                 "posword": "tab:red"}
    marker = {"nb201": "o", "transnas": "s", "nb101": "^"}
    models = [("pythia410m_step0", "Pythia-410M, random init"), ("qwen25_0.5b", "Qwen2.5-0.5B, trained")]
    dfs = {s: pd.read_csv(ROOT / "results" / "step3" / f"curves_{s}.csv") for s in marker}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True)
    for ax, (llm, title) in zip(axes, models):
        for space, df in dfs.items():
            mean = df[df["head"] == "rf"].groupby(["target", "features", "n_train"]).tau.mean()
            for fmt, c in fmt_color.items():
                if f"{fmt}_{llm}" not in mean.index.get_level_values("features"):
                    continue
                for (target, n), x in mean.xs(f"{fmt}_bow", level="features").items():
                    y = mean.get((target, f"{fmt}_{llm}", n))
                    if y is not None:
                        ax.scatter(x, y, color=c, marker=marker[space], s=14 + 10 * np.log10(n) ** 2,
                                   alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.plot([0, 0.85], [0, 0.85], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, 0.85), ax.set_ylim(0, 0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Kendall τ of token counts (bag of words)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall τ of LLM features")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=f) for f, c in fmt_color.items()]
    handles += [plt.Line2D([], [], color="gray", marker=m, linestyle="", label=s) for s, m in marker.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Word 排版小工具

CN_FONT, EN_FONT = "宋体", "Times New Roman"
HEAD_FONT = "黑体"


def set_font(run, size=None, bold=None, cn=CN_FONT, color=None):
    run.font.name = EN_FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), cn)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text="", size=10.5, bold=False, align=None, indent=True, space_after=4):
    """text 里用 **...** 标粗体。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.25
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align:
        p.alignment = align
    for k, piece in enumerate(text.split("**")):
        if piece:
            set_font(p.add_run(piece), size, bold or k % 2 == 1)
    return p


def bullet(doc, text, size=10.5):
    p = para(doc, "", size, indent=False, space_after=2)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    set_font(p.add_run("• "), size)
    for k, piece in enumerate(text.split("**")):
        if piece:
            set_font(p.add_run(piece), size, k % 2 == 1)
    return p


def heading(doc, text, level=1):
    size = {0: 18, 1: 14, 2: 12}[level]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level else 0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(text), size, True, HEAD_FONT)
    return p


def shade(cell, hex_color):
    tc_pr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def table(doc, rows, widths, size=9, bold_cells=()):
    """rows[0] 是表头；bold_cells 是要加粗的 (行, 列)。"""
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Cm(w)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            p = cell.paragraphs[0]
            p.paragraph_format.keep_with_next = i < len(rows) - 1  # 整张表放在同一页
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            if j > 0 and len(text) <= 8:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(text), size, i == 0 or (i, j) in bold_cells)
            if i == 0:
                shade(cell, "E8EEF4")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def caption(doc, text):
    p = para(doc, text, 9, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)
    return p


def figure(doc, path, width_cm, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, cap)


def callout(doc, text):
    """一格的浅色框，放一句话结论。"""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.cell(0, 0)
    shade(cell, "F3F6F9")
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.3
    for k, piece in enumerate(text.split("**")):
        if piece:
            set_font(p.add_run(piece), 10.5, k % 2 == 1)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# ---------------------------------------------------------------- 正文

def build():
    fig2 = HERE / "fig_scatter_compact.png"
    compact_scatter(fig2)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

    heading(doc, "阶段性进展汇报：冻结大模型特征用于 NAS 性能预测", 0)
    para(doc, "汇报人：__________　　　日期：2026 年 11 月 2 日", 10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
         indent=False, space_after=10)

    # ---- 0
    heading(doc, "一句话结论")
    callout(doc, "把神经网络结构写成代码、用冻结大模型（LLM）提取的平均池化特征，在 NAS 性能预测里表现得"
                 "**几乎就是代码文本的词频**（每种 token 出现了几次）。这一点同时解释了三个现象：小样本时比 one-hot "
                 "好、样本多时有一个算得出来的上限、预训练没有稳定的帮助。它在 3 个搜索空间、5 种文本写法、"
                 "Pythia 70M～1B 和 Qwen2.5-0.5B 上都成立；放到模拟搜索里反而是优点。")

    # ---- 1
    heading(doc, "1　背景与问题")
    para(doc, "近来不少工作把结构写成代码或文本，用 LLM 提取特征或直接打分来预测性能，例如 LLM-PP（Findings of "
              "ACL 2024）、COLE（GECCO 2026）、DR-LLM-ENAS（ESWA 2026）。这些工作报告了不错的结果，但"
              "**LLM 特征里到底有什么、预训练知识有没有起作用**，还没有人系统拆开看过。")
    para(doc, "按照师姐的建议，这一阶段先用小模型、在笔记本 CPU 上找能重复出现的规律，再决定是否需要更大的模型和显卡。")

    # ---- 2
    heading(doc, "2　实验设置")
    table(doc, [
        ["项目", "内容"],
        ["搜索空间", "NAS-Bench-201（全部 15625 个）；TransNAS-Bench-101 micro（全部 4096 个，7 个任务）；"
                   "NAS-Bench-101（随机 5000 个）"],
        ["LLM", "Pythia 70M / 160M / 410M / 1B，各取随机初始化和训练完成两个版本；Qwen2.5-0.5B。"
                "全部冻结，取最后一层隐藏状态的平均值"],
        ["对照特征", "one-hot（手工编码）；操作计数（5 个数）；代码文本的词频（bag of words）"],
        ["回归头", "岭回归、随机森林、MLP"],
        ["评估", "训练集 n = 10～2000，其余全部作测试集；Kendall τ-b 和 N@10；20 个随机种子，配对 Wilcoxon 检验"],
        ["硬件", "笔记本 CPU（Intel Core Ultra 5 125H），没有用显卡"],
    ], [2.6, 13.6])

    # ---- 3
    heading(doc, "3　主要发现")
    heading(doc, "3.1　主现象：小样本好，样本多有上限", 2)
    para(doc, "NAS-Bench-201、随机森林回归头：训练集只有 10 个结构时，随机初始化的 Pythia-410M 特征 τ=0.310，"
              "one-hot 只有 0.185；到 2000 个时反过来，LLM 特征停在 0.638，one-hot 涨到 0.772。另外两个搜索空间上"
              "现象相同（图 1）。")
    para(doc, "**解释：LLM 特征 ≈ 词频。** 15625 个结构按\"每种操作各用了几次\"只能分成 210 组，即使知道每组的真实"
              "平均精度，τ 最多也只有 0.629，这就是图中的上限虚线。证据有四条：(1) 只用 5 个操作计数就能复现小样本"
              "优势，而且更好；(2) 把操作随机换到别的边上，LLM 特征在小样本时几乎不受影响；(3) 线性探针显示随机 Pythia 特征的"
              "变化 75% 能被操作计数解释；(4) 直接用代码文本的词频（图中棕色虚线）和 LLM 特征几乎重合。")
    para(doc, "**预训练没有稳定的帮助：** 随机初始化的 Pythia 往往最好。有两个例外：TransNAS 上样本多时，"
              "以及 MLP 回归头样本多时，训练完成的模型更好。")
    figure(doc, ROOT / "results" / "step2" / "fig_summary_3spaces.png", 16.4,
           "图 1　三个搜索空间上的学习曲线（随机森林回归头）。黑线为 one-hot，蓝 / 红为随机 / 训练完成的 "
           "Pythia-410M，棕色虚线为代码词频，绿色点线为只靠操作计数的上限。")

    heading(doc, "3.2　四条一致的规律", 2)
    table(doc, [
        ["规律", "内容", "关键证据"],
        ["规律 2\n写法决定上限",
         "同一个结构换 5 种文本写法，LLM 特征的表现跟着该写法的词频走",
         "LLM τ 和词频 τ 的相关系数：随机 Pythia 0.93，训练完成 0.88，平均差距约 0.04（图 2）。"
         "只有\"位置 + 操作\"各用一个专门词的写法（posword）没有上限"],
        ["规律 3\n模型大小、家族不改变规律 2",
         "Pythia 从 70M 放大到 1B（14 倍），吻合程度不变；换 Qwen 也一样",
         "随机初始化 4 种大小都是 0.93；训练完成 0.86～0.88；Qwen2.5-0.5B 0.86"],
        ["规律 4\n换层、换取法去不掉上限",
         "第 6～24 层、平均或最后一个 token，上限都还在",
         "NB201 随机森林 n=1000：各层都在 0.61～0.66（one-hot 0.709）。探针能读出\"每条边放了什么\"（≈99%），"
         "但被计数成分淹没，回归头用不上"],
        ["规律 5\n对搜索是优点",
         "搜索时样本少，正好是词频类特征占优的范围",
         "200 次评估内找到真正第一名：随机 Pythia 68%，one-hot 32%（表 1）"],
    ], [3.4, 5.6, 7.2])
    figure(doc, fig2, 13.5,
           "图 2　每个点是一种\"写法 × 空间 × 任务 × 样本数\"组合：横轴为该写法词频的 τ，纵轴为 LLM 特征的 τ"
           "（随机森林）。点基本落在对角线上，说明 LLM 特征能做到的，词频也能做到。")

    heading(doc, "3.3　模拟搜索", 2)
    para(doc, "设置：先随机测 10 个结构，用随机森林预测器每轮再挑 10 个，测满 200 个为止；每种特征跑 50 次。"
              "下表为 50 次中找到池子里真正第一名的比例。随机 Pythia 在 5 个任务上都显著好于 one-hot"
              "（配对 Wilcoxon，p 在 2×10⁻⁵～0.003 之间）。")
    table(doc, [
        ["特征", "NB201", "NB101", "TransNAS\n场景", "TransNAS\n分割", "TransNAS\n自编码器", "平均"],
        ["随机搜索", "0%", "4%", "2%", "4%", "6%", "3%"],
        ["one-hot", "42%", "44%", "28%", "38%", "6%", "32%"],
        ["计数 + one-hot", "58%", "50%", "44%", "58%", "10%", "44%"],
        ["代码词频（不用 LLM）", "80%", "58%", "60%", "80%", "24%", "60%"],
        ["随机 Pythia-410M", "82%", "66%", "64%", "86%", "40%", "68%"],
        ["训练完成 Pythia-410M", "90%", "16%", "80%", "82%", "30%", "60%"],
        ["Qwen2.5-0.5B", "96%", "46%", "76%", "76%", "2%", "59%"],
    ], [4.4, 1.8, 1.8, 2.0, 2.0, 2.2, 1.8], bold_cells={(5, 6)})
    caption(doc, "表 1　200 次评估内找到最优结构的比例（50 次）。训练过的模型不稳定，例如 Qwen 在 NB201 上 96%，"
                 "在自编码器任务上只有 2%。")

    heading(doc, "3.4　和 COLE（GECCO 2026）的对照", 2)
    para(doc, "COLE 用冻结的 CodeLlama-7B 读 NB201 代码，报告 τ=0.797，高于我们算的上限 0.63，审稿人一定会问。"
              "我们照着 COLE 公开的代码复现了它的写法（每条边一个名字、跳连写成加法项），结果如下：")
    bullet(doc, "COLE 的写法把位置信息带进了词频，理论上限从 0.63 升到 0.70～0.72；")
    bullet(doc, "**但学出来的预测器没有变好**：同样的写法下，Pythia-410M 配三种回归头最高 0.651，和普通写法"
                "（0.650）一样。原因是位置信息混在数字 0～3 的出现次数里，2000 个样本学不出来；")
    bullet(doc, "所以 0.797 不是写法带来的。剩下最可能的原因是 COLE 的回归头（PCA 降到 128 维 + MLP + 排序损失），"
                "其次是 7B 模型（但 COLE 自己的消融显示模型大小影响很小）。0.797 和 one-hot 配随机森林的水平"
                "（0.778）相当。")

    # ---- 4
    heading(doc, "4　论文计划")
    para(doc, "**暂定题目：** What Do Frozen Language Models Read from a Neural Architecture? Mostly Token Counts")
    para(doc, "**贡献：** (1) 一个简单、可检验的解释：冻结 LLM 特征 ≈ 结构文本的词频；(2) 五条一致的规律，都指向"
              "\"写法决定上限，模型不决定\"；(3) 在 NAS 上第一次用随机初始化和训练存档系统拆出预训练的贡献；"
              "(4) 实用建议：做 LLM 预测器要报告词频和随机初始化基线；小样本优势用几十维的\"计数 + one-hot\""
              "就能拿到大部分。")
    para(doc, "**进度：** 实验数据已齐，提纲已写好（10 节，约 8 页），23 篇参考文献的出处已逐篇核对。"
              "**投稿方向想听老师的意见：** 先投 GECCO 或相关 workshop 的短文，补齐实验后再扩成期刊版"
              "（TEVC / SWEVO）。")

    # ---- 5
    heading(doc, "5　下一步与需要老师决定的事")
    para(doc, "投稿前还有下面几件事。实验部分做完这些就封顶，之后只写论文：")
    table(doc, [
        ["", "事项", "为什么要做", "资源", "时间（估计）"],
        ["A", "7B 级模型验证规律 2、3", "审稿人一定会问\"大模型还成立吗\"；COLE 用的就是 7B", "显卡", "约 1～2 天机时"],
        ["B", "按 COLE 的回归头重做对照", "分清 0.797 来自回归头还是模型", "CPU", "约半天"],
        ["C", "加一个专用预测器基线（GNN、CAP 或 DAP 之一）", "审稿人会问和专用方法比怎样", "CPU", "约 1～2 天"],
        ["D", "统计补齐：种子 20→30，多重比较 Holm 校正", "让显著性结论更稳", "CPU", "约 1 天"],
    ], [0.7, 4.4, 5.6, 1.4, 2.4])

    heading(doc, "显卡申请（事项 A）", 2)
    bullet(doc, "**用途：** 只做前向计算、提取特征，不训练模型。")
    bullet(doc, "**模型：** Pythia-6.9B（官方提供随机初始化和训练完成的存档，能和现有的 70M～1B 直接接上），"
                "再加一个代码模型，如 COLE 用的 CodeLlama-7B。")
    bullet(doc, "**数据量：** 3 个搜索空间约 1.4 万个结构 × 5 种写法，每段文本只有几百个 token。")
    bullet(doc, "**配置：** 一块显存 24GB 及以上的卡（如 RTX 3090 / 4090 / A5000），半精度即可放下 7B 模型；"
                "粗略估算每个模型版本约 5 小时，3 个版本（6.9B 随机 / 训练完成、代码模型）合计约 15 小时，留余量按 1～2 天。")
    bullet(doc, "**预期：** 按规律 3，7B 模型的点应该仍落在图 2 的对角线附近；如果不是，那本身就是一个重要发现。")

    heading(doc, "需要老师决定", 2)
    bullet(doc, "这个方向和结论是否值得按上面的提纲写成小论文，投哪里合适；")
    bullet(doc, "能否安排一块显卡（约 1～2 天）；")
    bullet(doc, "A～D 四件事做哪几件，其余的不再做。")

    para(doc, "附：完整结果、代码和提纲在项目目录 llm_demo/ 下（RESULTS.md、PAPER_OUTLINE.md、results/），"
              "可以随时提供。", 9, indent=False)

    doc.save(OUT)
    print(f"已保存 {OUT}")


if __name__ == "__main__":
    build()
