"""长任务用的简易进度条。

和常见的 tqdm 不同，它不在同一行反复刷新，而是每隔一段时间往下追加一行，
这样写进日志文件后，用 `Get-Content 日志.log -Wait -Encoding UTF8` 也能一行行看到进度。

用法:
    from progress import Progress
    bar = Progress(total=15625, name="pythia410m_step0", every=120)   # 每 120 秒记一行
    for batch in ...:
        ...
        bar.update(len(batch))                                         # 做完多少就加多少

输出示例:
    [pythia410m_step0] ██████░░░░░░░░░░░░░░  31%  4864/15625  已用 14m  剩约 31m  预计 23:12 完成
"""

import time
from datetime import datetime, timedelta


def _fmt(minutes):
    """把分钟数写成 1h05m 或 7m 这样的短格式。"""
    minutes = int(round(minutes))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


class Progress:
    def __init__(self, total, name="", every=120, width=20, units=True):
        """units=False 时不显示"已完成/总数"（总量是估出来的工作量、不是个数时用）。"""
        self.total, self.name, self.every, self.width, self.units = total, name, every, width, units
        self.done = 0
        self.note = ""
        self.start = self.last = time.time()
        self._print()  # 开始时先打一行 0%，表示已经在跑了

    def update(self, n=1, note=None):
        """做完 n 个单位就加 n；note 是附在行尾的说明（比如"现在在算哪一组"），可以不给。"""
        self.done += n
        if note is not None:
            self.note = note
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total - 1e-9:
            self.last = now
            self._print()

    def _print(self):
        frac = min(1.0, self.done / self.total) if self.total else 1.0
        full = int(frac * self.width)
        bar = "█" * full + "░" * (self.width - full)
        used = (time.time() - self.start) / 60
        line = f"[{self.name}] {bar} {frac * 100:4.0f}%"
        if self.units:
            line += f"  {self.done:.0f}/{self.total:.0f}"
        line += f"  已用 {_fmt(used)}"
        if 0 < self.done < self.total:  # 按目前的平均速度估算剩余时间
            left = used / self.done * (self.total - self.done)
            eta = (datetime.now() + timedelta(minutes=left)).strftime("%H:%M")
            line += f"  剩约 {_fmt(left)}  预计 {eta} 完成"
        elif self.done >= self.total - 1e-9:
            line += "  ✔ 完成"
        if self.note:
            line += f"  | {self.note}"
        print(line, flush=True)
