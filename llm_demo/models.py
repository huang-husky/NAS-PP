"""要比较的模型清单（下载脚本和特征提取脚本共用）。

名字 -> (HuggingFace 仓库, 版本)
Pythia 的 "stepN" 是训练到第 N 步时保存的存档；step0 就是还没训练过的随机初始化。
"""

from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def has_weights(model_dir):
    """模型文件夹里有没有完整的权重文件（大于 100MB 才算）。"""
    for f in ["pytorch_model.bin", "model.safetensors"]:
        p = Path(model_dir) / f
        if p.exists() and p.stat().st_size > 100e6:
            return True
    return False

MODELS = {
    # Pythia-410M 剂量曲线：同一个模型、同样的结构和分词器，只差"预训练了多少步"
    "pythia410m_step0": ("EleutherAI/pythia-410m", "step0"),
    "pythia410m_step1000": ("EleutherAI/pythia-410m", "step1000"),
    "pythia410m_step10000": ("EleutherAI/pythia-410m", "step10000"),
    "pythia410m_step143000": ("EleutherAI/pythia-410m", "step143000"),
    # 规律 3：不同大小的 Pythia（同一套训练数据和分词器），每个都比随机初始化和训练完成
    "pythia70m_step0": ("EleutherAI/pythia-70m", "step0"),
    "pythia70m_step143000": ("EleutherAI/pythia-70m", "step143000"),
    "pythia160m_step0": ("EleutherAI/pythia-160m", "step0"),
    "pythia160m_step143000": ("EleutherAI/pythia-160m", "step143000"),
    "pythia1b_step0": ("EleutherAI/pythia-1b", "step0"),
    "pythia1b_step143000": ("EleutherAI/pythia-1b", "step143000"),
    # Qwen2.5 对 Qwen2.5-Coder：同结构、同分词器，Coder 多学了大量代码
    "qwen25_0.5b": ("Qwen/Qwen2.5-0.5B", "main"),
    "qwen25coder_0.5b": ("Qwen/Qwen2.5-Coder-0.5B", "main"),
}
