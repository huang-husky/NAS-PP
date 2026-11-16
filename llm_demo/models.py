"""要比较的模型清单（下载脚本和特征提取脚本共用）。

名字 -> (HuggingFace 仓库, 版本)
Pythia 的 "stepN" 是训练到第 N 步时保存的存档；step0 就是还没训练过的随机初始化。
"""

from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def has_weights(model_dir):
    """Check readable weight containers; verify recorded checksums when available."""
    import json
    import zipfile
    model_dir = Path(model_dir)
    try:
        json.loads((model_dir / "config.json").read_text(encoding="utf8"))
        files = [p for p in [model_dir / "pytorch_model.bin", model_dir / "model.safetensors"] if p.exists()]
        if not files:
            return False
        for path in files:
            if path.suffix == ".bin":
                with zipfile.ZipFile(path) as archive:
                    if not any(n.endswith("data.pkl") for n in archive.namelist()):
                        return False
            else:
                from safetensors import safe_open
                with safe_open(path, framework="np") as archive:
                    if not list(archive.keys()):
                        return False
        provenance = model_dir / "provenance.json"
        if provenance.exists():
            from artifacts import file_hash
            recorded = json.loads(provenance.read_text(encoding="utf8"))["weights"]
            if any(recorded.get(p.name) != file_hash(p) for p in files):
                return False
        return True
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
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
