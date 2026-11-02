# 论文提纲（草稿 v1，2026-11-01）

> 所有数字都来自 `RESULTS.md`，写正文时再逐个回到 `results/` 核对。
> 参考文献的出处（作者、会议或期刊、arXiv 号）已于 11-01 在网上逐篇核对，见文末"参考文献核对表"。

---

## 题目（候选）

1. **What Do Frozen Language Models Read from a Neural Architecture? Mostly Token Counts**
2. Frozen LLM Features for NAS Performance Prediction Are Token Counts in Disguise
3. 中文：冻结大模型读神经网络结构时读到了什么？——以 NAS 性能预测为例

倾向第 1 个：问题式题目，结论直接放在副标题里。

## 一句话

平均池化的冻结 LLM 特征，在 NAS 性能预测里表现得几乎就是"结构文本的词频"。这一点同时解释了三件事：为什么小样本好、为什么样本多有上限、为什么预训练没用。它跨 3 个搜索空间、5 种写法、5 种模型大小、2 个模型家族、各层和两种取法都成立，而且在实际搜索中是优点。

## 摘要（草稿）

近来不少工作把神经网络结构写成代码或文本，用大语言模型（LLM）提取特征来预测结构性能，但这些特征里到底有什么还不清楚。我们在 NAS-Bench-201、NAS-Bench-101 和 TransNAS-Bench-101 三个搜索空间上做了系统的对照实验。主要发现是：平均池化的冻结 LLM 特征，表现得几乎就是结构文本的词频（每种 token 出现了几次）。这一点解释了三个现象。第一，小样本时 LLM 特征明显好于 one-hot 编码，因为词频把结构分成少量的组，容易学。第二，样本多时有一个算得出来的上限，因为同一组里的结构分不开。第三，预训练没有带来稳定的好处，随机初始化的模型往往最好。我们进一步发现：(1) LLM 特征的表现由文本写法决定，在 5 种写法上和词频的相关系数为 0.86～0.93；(2) 这个规律在 Pythia 70M～1B 和 Qwen2.5-0.5B 上都成立，不随模型大小变化；(3) 换哪一层、用哪种取法都去不掉上限，位置信息能被探针读出，但回归头用不上；(4) 在模拟搜索中，这一特性反而是优点：200 次评估内，随机初始化 LLM 特征找到最优结构的比例为 68%，one-hot 为 32%。这些结果说明，要改进 LLM 性能预测器，该改的是结构的写法，而不只是换更大的模型。

## 贡献

1. **一个简单、可检验的解释。** 冻结 LLM 特征 ≈ 结构文本的词频（加一层随机的非线性混合）。我们用算得出的上限、打乱位置、线性探针和一个事先定好的词频检验来验证它。
2. **五条一致的规律**，都指向同一个结论：写法决定上限，模型大小、模型家族、层和取法都不决定。
3. **NAS 领域第一次系统拆出预训练的贡献。** 我们用 Pythia 训练过程中的存档和随机初始化做对照，发现预训练在小样本时没有帮助。一般回归任务上的结论是"预训练通常有帮助"（Tang 等，TMLR 2025），我们在 NAS 上看到的不一样。
4. **实用建议。** 小样本优势不需要 LLM，"计数 + one-hot"这类几十维特征就能拿到大部分；做搜索时（样本少），词频类特征比 one-hot 更容易找到最优结构。

---

## 正文结构

### 1 引言（约 1 页）

- 背景：NAS 很贵，性能预测器是主要的加速手段（E2EPP；Song 等的综述）。
- 新趋势：把结构写成代码或文本，让 LLM 提取特征或直接打分（Jawahar；COLE；DR-LLM-ENAS 等）。
- 缺口：这些工作报告了"好用"，但没人回答 LLM 特征里到底有什么，也没人严格拆出预训练的贡献。
- 我们的出发点：一个奇怪的现象。小样本时 LLM 好，样本多时被 one-hot 反超，而且随机初始化的模型最好。
- 主要发现和贡献（对应上一节）。
- **图 1**：主现象。学习曲线：one-hot 对比随机 / 训练完成 Pythia，用 NB201、随机森林（`results/fig1_tau_vs_n.png`，可能要重画成单栏）。

### 2 相关工作（约 0.75 页）

（每篇的出处见文末"参考文献核对表"。）

**2.1 NAS 性能预测器。**
- E2EPP：随机森林性能预测器（Sun 等，TEVC 2020）。这是组内工作，也是我们回归头选随机森林的依据。
- 高效评估方法综述（Song 等，组内）。
- White 等（NeurIPS 2021）：第一次大规模比较了 31 种性能预测器。
- 小样本、迁移类预测器（都是组内工作）：
  - CAP（Ji、Feng、Sun，IJCAI 2024）：自监督预训练，NAS-Bench-101 上只用 172 个标注结构；
  - DAP（Song 等，IEEE TC）：把已有基准上训练的预测器迁移到新空间。
- 写法：这些是"专门为小样本设计"的预测器，正好是我们的强基线（见"投稿前还缺什么"第 2 条）。

**2.2 LLM 做 NAS 生成和搜索。**
- EvoPrompting、GENIUS、LLMatic、GPT-NAS、LLM Guided Evolution、GNAS-LLM（GPT4GNAS）。
- 相邻领域的结构搜索：tnGPS（张量网络结构）、ReStruct（元结构）。
- 这一支用 LLM 来"提出结构"，和我们关注的"评价结构"不同，一段带过即可。

**2.3 LLM 做性能预测器（最相关）。**
- 直接提示：
  - Jawahar 等（Findings of ACL 2024）：GPT-4 直接打分，再蒸馏成小 MLP；
  - DR-LLM-ENAS（Xu 等，ESWA 2026）：先比较、再打分。
- 冻结编码 + 回归头：**COLE（Somu 等，GECCO 2026）**，已读全文（arXiv 扩展版）。
  - 设置：
    - PyTorch 代码 → 冻结的 CodeLlama-Python-7B → 最后一层平均池化；
    - PCA 降到 128 维 → 3 层 MLP，用成对排序损失；
    - NAS-Bench-201 和 einspace，N = 14～3516。
  - NB201 上 τ 从 0.463（N=14）升到 0.797（N=3516）。**没有出现我们看到的上限**（见下面"必须先回答的问题"）。
  - 几条结果和我们的规律 3 一致：
    - 34B 比 7B 只好一点点；
    - Yi-Coder 1.5B 在 N=3516 时还略好于 9B（0.791 对 0.779）；
    - CodeLlama 34B 基础版和 Python 版差别不超过 0.008。
  - 代码输入比 ONNX 文本好（N≥220 时显著）。
  - BANANAS 搜索：CIFAR-100 上达到最优 1% 以内少用 34% 的评估，CIFAR-10 少用 13%，ImageNet16 少用 18%。
  - 它没做的：随机初始化对照、词频基线、打乱对照、池化方式比较；也没写清楚预测的是哪个数据集的精度。
  - **我们的工作正好回答它没回答的问题：这些特征里到底是什么。**
- 微调语言模型：
  - Transferrable Surrogates（Qin 等，AutoML 2025）：微调 ModernBERT 等模型，空间为 einspace；
  - RLM for Code（Akhauri 等，ICML 2026）：5 个 NAS 空间平均 Kendall τ 0.46；
  - ONNX-Net（Qin 等，arXiv 2025）：把 ONNX 图写成文本，做跨空间的通用预测器。
- 排序式：LM-Searcher（Hu 等，EMNLP 2025）。
- 共同点：都没有用"同一个模型的随机初始化版本"做对照，所以分不清"预训练知识"和"特征形式"各贡献了多少。

**2.4 随机编码器和 LLM 嵌入做回归。**
- Wieting & Kiela（ICLR 2019）：随机权重的句子编码器是很强的基线，作者认为性能主要来自词向量，而不是编码器。**这和我们"主要是词频"的结论是同一个方向，要重点引用。**
- Pilault 等（ACL 2019 学生研讨会）：摘要任务里，随机编码器和训练过的编码器差不多。
- Tang 等（TMLR 2025），Understanding LLM Embeddings for Regression，已读全文：
  - 设置：
    - 模型：冻结的 T5（Small～XXL）和 Gemini 1.0；平均池化（他们发现平均池化最好，最后一个 token 最差）；
    - 回归头：**只用 MLP**（两层，宽 256）；
    - 输入：超参数这类**数值型**键值对字符串，例如 `{lr: 0.01, ...}`；
    - 任务：合成函数 BBOB，以及 Google Vizier 的真实调参任务。
  - 预训练、随机初始化、只用词表嵌入三者比较：
    - 预训练通常更好，T5 越大越明显；
    - 但在 AutoML、L2DA 两类真实任务上提升"很小"；
    - Gemini 更大不一定更好。
  - 真实任务上，平均来说 LLM 特征还略差于传统特征。他们用"嵌入空间更平滑"来解释 LLM 特征为什么在高维时有优势。
  - **和我们不矛盾，反而对得上：**
    1. 他们只用 MLP。我们用 MLP 时，样本多时预训练也有帮助（NB201 上 n=2000 时 +0.133）；
    2. 他们的输入是连续数值，我们的是少数几个离散的操作名；
    3. 平均池化好于最后一个 token，和我们规律 4 的第 4 条一致。
- 我们的结论在 NAS 上更进一步：用随机森林和岭回归时，预训练在小样本时没有帮助，甚至更差。第 9 节解释原因。

### 3 实验设置（约 1 页）

- **搜索空间**：
  - NB201：15625 个结构，标签为 CIFAR-10 / 100 / ImageNet16。
  - TransNAS-Bench-101 micro：4096 个结构，7 个任务。
  - NB101：从 42 万个里抽 5000 个。
- **写法**：默认为 PyTorch 风格代码，规律 2 里扩展到 5 种（表 1）。
- **模型**：
  - Pythia 70M / 160M / 410M / 1B，每种都有第 0 步（随机初始化）和第 143000 步（训练完成）。
  - Pythia-410M 的第 1000、10000 步存档。
  - Qwen2.5-0.5B 和 Qwen2.5-Coder-0.5B。
- **特征**：最后一层平均池化（默认）；规律 4 中换层、换取法。
- **对照特征**：
  - one-hot；
  - 操作计数；
  - 计数 + one-hot；
  - 词频（用模型自己的分词器，事先定好、不挑词）。
- **回归头**：岭回归、随机森林（主要）、MLP。
- **指标和统计**：
  - Kendall τ-b、N@10；
  - 训练集 n = 10～2000，其余全部作测试集；
  - 20 个种子，同一批划分上用配对 Wilcoxon 检验。

### 4 主现象和解释：LLM 特征 ≈ 词频（约 1.5 页，全文核心）

1. **现象**（NB201）：
   - n=10 时，随机 Pythia 比 one-hot 高 0.125（随机森林，p<0.001）；
   - n=2000 时反而低 0.134；
   - 训练完成的模型不比随机的好。
2. **算得出的上限**：
   - 15625 个结构只有 210 种操作计数，τ 最多 0.629，LLM 停在 0.63～0.64。
   - NB101 上换成 6 个数（含连线计数），上限 0.625，LLM 为 0.603。
3. **打乱位置**：
   - LLM 特征小样本时不受影响（0.322 对 0.310）；
   - one-hot 在 n=2000 时掉 0.22。
4. **线性探针**：随机 Pythia 特征的变化中，75% 能被计数线性解释。
5. **事先定好的词频检验**：
   - 三个空间上都复现了小样本优势；
   - 样本多时，LLM 只比词频多 0.01～0.06。
6. **跨空间**：TransNAS 7 个任务重复了同样的模式。自编码器是"反例"：计数最没用的任务上，LLM 的小样本优势也消失了。
- **图 2**：三个空间汇总（`results/step2/fig_summary_3spaces.png`）。
- **表 2**：上限对比，即计数 / 词频上限和 LLM 在最大 n 时的成绩。

### 5 规律 2：上限由写法决定，不由模型决定（约 1 页）

- 5 种写法：code、edges、sorted、nl、posword（表 1 给例子）。
- LLM τ 和词频 τ 的相关系数：
  - 随机 Pythia 为 0.93；
  - 训练完成的为 0.88；
  - 平均差距约 0.04。
- "只列操作"（sorted）：删掉位置信息后，LLM 和词频一模一样。文本里没有的信息，模型也变不出来。
- posword 写法：
  - 这种写法的词频就等于 one-hot；
  - 小样本优势消失，大样本最好；
  - 配随机森林时仍追不上 one-hot。原因是随机旋转实验：one-hot 被随机旋转后，随机森林从 0.71 掉到 0.56，岭回归不变。
- **图 3**：散点图，横轴为词频 τ，纵轴为 LLM τ（`results/step3/fig_formats_scatter.png`）。

### 6 规律 3：模型大小和家族都不改变规律 2（约 0.5 页）

- Pythia 70M～1B：
  - 随机初始化的模型，相关系数都是 0.93；
  - 训练完成的为 0.86～0.88。
- Qwen2.5-0.5B 为 0.86，和训练完成的 Pythia 相同。
- **图 4**：`results/step3/fig_sizes.png`。
- 这一节为"用更大模型验证"提供了明确的预期，引出局限和下一步。

### 7 规律 4：换层、换取法都去不掉上限（约 0.5 页）

- 第 0 层取平均就是词频。
- 第 6～24 层、平均或最后一个 token，上限都还在。
- 位置信息"读得出、用不上"：探针能读出 99%，但预测不变好，因为它被计数成分淹没了。
- **图 5**：`results/step5/fig_layers.png`。

### 8 规律 5：对搜索的影响（约 0.75 页）

- N@10 也跟着词频走：随机 Pythia 的相关系数为 0.89。
- 模拟搜索：
  - 设置：随机森林预测器，每轮测 10 个，预算 200，50 次。
  - 找到第一名的平均比例：随机 Pythia 68%，词频 60%，训练完成的 Pythia 60%，Qwen 59%，one-hot 32%，随机搜索 3%。
  - 5 个任务上，随机 Pythia 都显著好于 one-hot。
- 训练过的模型不稳定。例如 Qwen 在 NB201 上为 96%，在自编码器上为 2%。
- **图 6**：`results/step6/fig_search.png`。
- **表 3**：搜索成功率。

### 9 讨论（约 0.75 页）

- **为什么和 Tang 等（一般回归）的结论不同？** NAS 结构文本的词表极小、格式极整齐，会变的只有少数几个词。平均池化之后，词频就成了主要成分，预训练知识没有地方发挥。这也和 Wieting & Kiela"性能主要来自词向量、不来自编码器"的发现一致。
- **预训练什么时候有用？** 有两个例外，要诚实写出来：
  - TransNAS 上样本多时（随机森林，7 个任务全部更好）；
  - MLP 回归头、样本多时。
  可能的原因是"计数以外的成分"在上限低、样本多时才开始有用。
- **给做 LLM 预测器的人的建议**：
  1. 一定要报告词频 / 计数基线和随机初始化对照；
  2. 想突破上限，就改写法，让位置信息以**单独的词**进入词频（像 posword；像 COLE 那样混在共享词里没有用）；
  3. 小样本场景可以直接用"计数 + one-hot"。
- **局限**：
  - 模型最大只到 1B；
  - 只研究了平均池化的冻结特征，没有研究微调和提示式预测；
  - NB101 只用了 5000 个结构；
  - 连线计数是看过结果后挑的（已用事先定好的词频检验补救）；
  - 没有和 GNN、CAP、DAP 等专用预测器比；
  - NB201 的同构结构没有去重。

### 10 结论（约 0.25 页）

---

## 图表清单

| 编号 | 内容 | 现有文件 | 要做的 |
|---|---|---|---|
| 图 1 | 主现象学习曲线 | `results/fig1_tau_vs_n.png` | 精简成单栏（只留随机森林、3～4 条线） |
| 图 2 | 三个空间汇总 | `results/step2/fig_summary_3spaces.png` | 可直接用 |
| 图 3 | 写法散点图 | `results/step3/fig_formats_scatter.png` | 正文只放 410M 和 Qwen，其余放附录 |
| 图 4 | 模型大小 | `results/step3/fig_sizes.png` | 可直接用 |
| 图 5 | 层 × 取法 | `results/step5/fig_layers.png` | 可直接用 |
| 图 6 | 模拟搜索 | `results/step6/fig_search.png` | 图例太小，要重画 |
| 表 1 | 5 种写法例子 | `RESULTS.md` 规律 2 | 整理成英文 |
| 表 2 | 上限对比 | `RESULTS.md` 解释实验、NB101、词频 | 汇总成一张表 |
| 表 3 | 搜索成功率 | `RESULTS.md` 规律 5 | 可直接用 |
| 可选 | 示意图：结构 → 代码 → LLM → 平均池化 ≈ 词频 | 无 | 新画，放引言 |

## COLE 问题：已做第一轮检验（2026-11-01），还没完全回答

COLE 在 NB201 上 N=3516 时 τ=0.797，远高于我们的计数上限 0.63。审稿人一定会问。详细结果见 `RESULTS.md`"对照检验：COLE 的写法"。

**已经知道的：**
- **写法：** COLE 的写法（每条边一个名字、跳连写成加法项）确实把位置信息带进了词频，组内平均上限从 0.63 升到 0.70～0.72。
- **但学不出来：** 用同样的写法，Pythia-410M 配随机森林 / 岭回归 / MLP，n=2000 最高 0.651，和普通 code 写法一样（词频也一样，0.617 对 0.622）。原因是位置信息混在数字 `0`～`3` 的次数里，分成 3536 组后每组太小，样本不够学。
- **规律 2 在 cole 写法下仍然成立**（NB201 上 LLM 和词频差 −0.05～+0.04）。
- **0.797 本身** 和 one-hot 配随机森林差不多（n=2000 时 0.778），并没有超出"结构信息本身"能给的范围。

**还剩下的解释：**
- 回归头（PCA 128 维 + MLP + 成对排序损失），这是最可能的；
- 7B 模型（但 COLE 自己的消融说大小影响很小）；
- 评估方式（数据集不明、10 折分层、N 更大）。

**下一步（CPU 可做，约半天）：** 照着 COLE 的回归头，在 cole 写法的 Pythia 特征、cole 词频、one-hot 上各训练一次：
- 如果三者都升到 0.75 以上：说明是回归头的功劳，不是 LLM 的功劳。规律 2 仍然成立，可以写进正文当"外部验证"。
- 如果只有 LLM 特征升上去：说明好的回归头能用上 LLM 特征里"计数以外"的东西。结论要收窄成"在常用回归头下"。

**写进论文时：** 放在第 5 节末尾或讨论里，作为对规律 2 的修正：位置信息要以单独的词进入词频（像 posword）才有用；混在共享词的次数里（像 COLE），上限变高，但学不出来。

## 投稿前还缺什么（按重要性）

0. **COLE 问题（写法部分已做完）。** 写法解释不了 0.797；还要照着 COLE 的回归头重做一次，分清是回归头还是模型的功劳。CPU 约半天，最好在找老师之前做完。

1. **更大的模型**（7B 左右，需要显卡）：验证规律 2、3 在大模型上是否仍然成立。这也是向孙老师申请显卡的理由。
2. **专用预测器基线**（GNN 类、CAP 或 DAP 至少一个）：审稿人一定会问。
3. **统计上补齐**：
   - 种子从 20 加到 30；
   - 多重比较用 Holm 校正。
4. **（已完成）两篇关键文献已读全文**：COLE 和 Tang 等的设置和数字已写进第 2 节；DAP 的年份和 DOI 还要向组里确认。
5. （可选）NB101 用更多结构；加一个新搜索空间（如 NAS-Bench-Graph）。

## 投稿方向（供讨论，听老师意见）

调研文档里的建议：实验扎实、结论清晰时，投 TEVC / SWEVO / GECCO。先做成一篇"小论文"的话，可以考虑：
- GECCO 或相关 workshop（篇幅短，适合"一个清晰的发现"）；
- 补齐上面 1～3 后再扩成期刊版。

## 参考文献核对表（2026-11-01 核对）

"本地"表示 PDF 在 `essays/` 里；"网上"表示在 arXiv、会议官网或出版方页面上核对过作者和出处。

| 简称 | 题目 | 作者 | 出处 | 核对 |
|---|---|---|---|---|
| E2EPP | Surrogate-Assisted Evolutionary Deep Learning Using an End-to-End Random Forest-Based Performance Predictor | Sun, Wang, Xue, Jin, Yen, Zhang | IEEE TEVC 24(2), 2020 | 本地 |
| EEM 综述 | Efficient Evaluation Methods for Neural Architecture Search: A Survey | Song, Xie, Lv, Yen, Ding, Lv, Sun | arXiv 2301.05919 | 本地 |
| White 等 | How Powerful are Performance Predictors in Neural Architecture Search? | White, Zela, Ru, Liu, Hutter | NeurIPS 2021（arXiv 2104.01177） | 网上 |
| CAP | CAP: A Context-Aware Neural Predictor for NAS | Ji, Feng, Sun | IJCAI 2024（arXiv 2406.02056） | 网上 |
| DAP | DAP: Domain Adaptive Performance Predictor for Efficient Neural Architecture Search | Song, Xie, Feng, Song, Sun | IEEE Transactions on Computers（ResearchGate 显示 2025 年 1 月） | 网上；**年份和 DOI 不一致，引用前去 IEEE Xplore 确认，或直接问组里** |
| EvoPrompting | EvoPrompting: Language Models for Code-Level Neural Architecture Search | Chen, Dohan, So | arXiv 2302.14838 | 本地 |
| GENIUS | Can GPT-4 Perform Neural Architecture Search? | Zheng 等 | arXiv 2304.10970 | 本地 |
| GPT-NAS | GPT-NAS: Evolutionary Neural Architecture Search with the Generative Pre-Trained Model | Yu 等 | arXiv 2305.05351 | 本地 |
| LLMatic | LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization | Nasir 等 | arXiv 2306.01102 | 本地 |
| Guided Evolution | LLM Guided Evolution - The Automation of Models Advancing Models | Morris, Jurado, Zutty | arXiv 2403.11446 | 本地 |
| GNAS-LLM | Graph Neural Architecture Search with Large Language Models | Wang 等 | Science China Information Sciences（arXiv 2310.01436） | 本地 |
| tnGPS | tnGPS: Discovering Unknown Tensor Network Structure Search Algorithms via LLMs | Zeng 等 | arXiv 2402.02456 | 本地 |
| ReStruct | Large Language Model-driven Meta-structure Discovery in Heterogeneous Information Network | Chen 等 | arXiv 2402.11518 | 本地 |
| LLM-PP | LLM Performance Predictors are good initializers for Architecture Search | Jawahar, Abdul-Mageed, Lakshmanan, Ding | Findings of ACL 2024（arXiv 2310.16712） | 本地 + 网上 |
| DR-LLM-ENAS | Dual-role LLMs for evolutionary neural architecture search | Xu, Fang, Xue | Expert Systems With Applications 330, 2026 | 本地 |
| COLE | Towards Code-Oriented LM Embeddings for Surrogate-Assisted Neural Architecture Search | Somu, Balakrishnan, Kravtsov, McDaniel, Zutty | GECCO 2026（arXiv 2605.15649 是扩展版） | 网上，已读全文（arXiv HTML 版） |
| Transferrable Surrogates | Transferrable Surrogates in Expressive Neural Architecture Search Spaces | Qin, Kadlecová, Pilát 等 | AutoML 2025（PMLR 293；arXiv 2504.12971） | 网上 |
| RLM for Code | Regression Language Models for Code | Akhauri, Song, Wongpanich, Lewandowski, Abdelfattah | ICML 2026（PMLR 306；arXiv 2509.26476） | 网上 |
| ONNX-Net | ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures | Qin, Auras, Cohen, Crowley | arXiv 2510.04938 | 网上；没查到正式发表的会议 |
| LM-Searcher | LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding | Hu 等 | EMNLP 2025（arXiv 2509.05657） | 网上 |
| Wieting & Kiela | No Training Required: Exploring Random Encoders for Sentence Classification | Wieting, Kiela | ICLR 2019（arXiv 1901.10444） | 网上 |
| Pilault 等 | On the Impressive Performance of Randomly Weighted Encoders in Summarization Tasks | Pilault, Park, Pal | ACL 2019 学生研讨会（arXiv 2002.09084） | 网上 |
| Tang 等 | Understanding LLM Embeddings for Regression | Tang, Yang, Song | TMLR 2025（arXiv 2411.14708） | 网上，已读全文（arXiv HTML 版） |

数据集（写设置时引用）：NAS-Bench-101（Ying 等，ICML 2019）、NAS-Bench-201（Dong & Yang，ICLR 2020）、TransNAS-Bench-101（Duan 等，CVPR 2021）；模型：Pythia（Biderman 等，ICML 2023）、Qwen2.5（Qwen 团队技术报告，2024）。这几篇是领域里的标准引用，写正文时再补全页码。
