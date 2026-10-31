"""第 0 步：把要用的 6 个模型下载到 llm_demo/models/ 下（只需要跑一次）。

运行方式（在 llm_demo 文件夹下）:
    python download_models.py

已经下载完的模型会自动跳过，所以中途断了可以直接重新运行。
"""

import os
import shutil
import time
from pathlib import Path

# 必须在导入 huggingface_hub 之前设置：
# 1. 关掉新的 Xet 下载通道，它在这台电脑上会无声地挂住；改用普通 HTTP 下载
# 2. 60 秒收不到数据就算超时，交给下面的重试逻辑，而不是一直卡着
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from huggingface_hub import snapshot_download  # noqa: E402

from models import MODELS, MODEL_DIR, has_weights  # noqa: E402

# 只下载这些文件。Pythia 的存档里同时有 .bin（约 0.9GB）和 .safetensors（约 1.6GB），
# 两者内容相同，只要小的那份 .bin；Qwen 只有 .safetensors。
PATTERNS = ["*.json", "*.txt", "pytorch_model.bin", "model.safetensors"]

for name, (repo, revision) in MODELS.items():
    target = MODEL_DIR / name
    done_flag = target / ".download_done"
    if done_flag.exists() and has_weights(target):
        print(f"[跳过] {name} 已下载")
        continue

    print(f"[下载] {name}  ({repo} @ {revision})", flush=True)
    patterns = PATTERNS
    if repo.startswith("EleutherAI/pythia"):
        patterns = [p for p in PATTERNS if p != "model.safetensors"]
    for attempt in range(1, 6):  # 断线或超时就重试，最多 5 次
        try:
            snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=patterns)
            # 断网时 snapshot_download 可能不报错、直接返回空文件夹，所以要亲自确认权重文件在
            if has_weights(target):
                break
            raise RuntimeError("权重文件没有下载下来")
        except Exception as e:  # noqa: BLE001
            print(f"  第 {attempt} 次失败：{type(e).__name__}，10 秒后重试", flush=True)
            time.sleep(10)
    else:
        raise SystemExit(f"{name} 下载失败 5 次，请检查网络后重新运行")
    shutil.rmtree(target / ".cache", ignore_errors=True)  # 下载工具留下的临时记录，用不到
    done_flag.touch()
    size_gb = sum(f.stat().st_size for f in Path(target).rglob("*") if f.is_file()) / 1e9
    print(f"[完成] {name}  {size_gb:.2f} GB", flush=True)

print("全部模型下载完成")
