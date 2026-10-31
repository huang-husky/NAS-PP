﻿# 全空间实验进度日志：每 2.5 分钟往下追加一行，模型算完时额外记一行。
# 只读取 features/ 里的文件和 run_full.log，不改任何东西；按 Ctrl+C 关掉不影响实验。

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$feat     = Join-Path $dir 'features'
$log      = Join-Path $dir 'run_full.log'
$models   = @('pythia410m_step0', 'pythia410m_step1000', 'pythia410m_step10000',
              'pythia410m_step143000', 'qwen25_0.5b', 'qwen25coder_0.5b')
$interval = 150     # 每隔多少秒记一行
$guessMin = 45      # 第一个模型算完前，先按每个模型 45 分钟估算
$evalTotal = 12     # 评估一共 12 组：2 个目标 x 6 种训练样本数
# 每组评估预计分钟数（训练样本 n = 10/50/100/500/1000/2000，两个目标各一遍）。
# 训练样本越多越慢：前 5 组是实测的；n 翻倍时间约翻 3 倍，n=2000 按 65 分钟估算
$evalGroupMin = @(2, 3, 2, 8, 23, 65, 2, 3, 2, 8, 23, 65)
$evalMin  = ($evalGroupMin | Measure-Object -Sum).Sum

function Bar([double]$frac, [int]$width) {
    $frac = [math]::Max(0.0, [math]::Min(1.0, $frac))  # 写成 1.0 而不是 1，否则 PowerShell 会把小数按整数算
    $full = [int][math]::Floor($frac * $width)
    return ('█' * $full) + ('░' * ($width - $full))
}

function Fmt-Min([double]$m) {
    if ($m -ge 60) { return '{0}h{1:00}m' -f [int][math]::Floor($m / 60), [int]($m % 60) }
    return '{0}m' -f [int]$m
}

$start = (Get-Item (Join-Path $feat 'pool.npy')).LastWriteTime
$seen = @{}

Write-Host ''
Write-Host '  NAS-Bench-201 全空间实验 · 进度日志' -ForegroundColor Cyan
Write-Host ('  ' + ('─' * 70)) -ForegroundColor DarkGray
Write-Host ("  开始于 {0}   6 个模型提取特征 → 训练和评估   每 {1} 分钟记一行" -f $start.ToString('HH:mm'), ($interval / 60)) -ForegroundColor DarkGray
Write-Host '  关掉这个窗口或按 Ctrl+C 都不影响实验' -ForegroundColor DarkGray
Write-Host ('  ' + ('─' * 70)) -ForegroundColor DarkGray
Write-Host ''

while ($true) {
    $now = Get-Date
    $logText = if (Test-Path $log) { Get-Content $log -Encoding UTF8 -Raw } else { '' }

    # 已经算完的模型，以及每个用了多久
    $prev = $start; $durations = @(); $current = $null; $currentIdx = 0
    for ($i = 0; $i -lt $models.Count; $i++) {
        $f = Join-Path $feat "$($models[$i]).npy"
        if (Test-Path $f) {
            $t = (Get-Item $f).LastWriteTime
            $d = ($t - $prev).TotalMinutes
            $durations += $d
            if (-not $seen.ContainsKey($models[$i])) {
                $seen[$models[$i]] = $true
                Write-Host ("  ✔ {0}  {1} 特征算完，用时 {2}" -f $t.ToString('HH:mm'), $models[$i], (Fmt-Min $d)) -ForegroundColor Green
            }
            $prev = $t
        } elseif (-not $current) {
            $current = $models[$i]; $currentIdx = $i + 1
        }
    }
    $done = $durations.Count
    # 有实测数据后，就用实测的平均用时来估算
    $perModel = if ($done -gt 0) { ($durations | Measure-Object -Average).Average } else { $guessMin }
    $totalMin = $models.Count * $perModel + $evalMin
    $evalDone = ([regex]::Matches($logText, '\[完成\] 目标')).Count

    if ($current) {
        $el = ($now - $prev).TotalMinutes
        $curFrac = [math]::Min(0.99, $el / $perModel)
        $doneMin = $done * $perModel + $curFrac * $perModel
        $stage = '特征 {0}/6  {1,-21}' -f $currentIdx, $current
    } else {
        $evalDoneMin = if ($evalDone -gt 0) { ($evalGroupMin[0..($evalDone - 1)] | Measure-Object -Sum).Sum } else { 0 }
        # 日志最后一次写入 = 上一组跑完的时间，用它算当前这组已经跑了多久，让进度在组内也能往前走
        if ($evalDone -lt $evalTotal) {
            $groupEl = ($now - (Get-Item $log).LastWriteTime).TotalMinutes
            $evalDoneMin += [math]::Min($groupEl, 0.95 * $evalGroupMin[$evalDone])
        }
        $curFrac = $evalDoneMin / $evalMin
        $doneMin = $models.Count * $perModel + $evalDoneMin
        $stage = '评估 第{0,2}/{1} 组{2}' -f [math]::Min($evalDone + 1, $evalTotal), $evalTotal, (' ' * 16)
    }
    $leftMin = [math]::Max(0, $totalMin - $doneMin)
    $allFrac = $doneMin / $totalMin

    # 跑完 / 出错 / 程序消失，都单独说清楚然后停下
    if ($logText -match 'EXIT 0') {
        Write-Host ''
        Write-Host ("  ★ {0}  全部跑完！总用时 {1}，结果在 results 文件夹" -f $now.ToString('HH:mm'), (Fmt-Min ($now - $start).TotalMinutes)) -ForegroundColor Green
        break
    }
    if ($logText -match 'Traceback|EXIT [1-9]') {
        Write-Host ''
        Write-Host ("  ✘ {0}  日志里有报错，把 run_full.log 发给 Claude 看看" -f $now.ToString('HH:mm')) -ForegroundColor Red
        break
    }
    if (-not (Get-Process python -ErrorAction SilentlyContinue)) {
        Write-Host ("  ! {0}  没找到实验程序（python），可能停了，告诉 Claude 看看" -f $now.ToString('HH:mm')) -ForegroundColor Red
    }

    # 一行进度：时间  总进度条  百分比 │ 当前阶段  当前进度条  百分比 │ 剩余时间
    Write-Host ('  {0}  ' -f $now.ToString('HH:mm')) -NoNewline -ForegroundColor DarkGray
    Write-Host (Bar $allFrac 20) -NoNewline -ForegroundColor Cyan
    Write-Host (' {0,3}%' -f [int](100 * $allFrac)) -NoNewline -ForegroundColor White
    Write-Host '  │  ' -NoNewline -ForegroundColor DarkGray
    Write-Host $stage -NoNewline -ForegroundColor Gray
    Write-Host (' ' + (Bar $curFrac 10)) -NoNewline -ForegroundColor Yellow
    Write-Host (' {0,3}%' -f [int](100 * $curFrac)) -NoNewline -ForegroundColor White
    Write-Host '  │  ' -NoNewline -ForegroundColor DarkGray
    Write-Host ('剩约 {0}，{1} 左右完成' -f (Fmt-Min $leftMin), $now.AddMinutes($leftMin).ToString('HH:mm')) -ForegroundColor DarkGray

    Start-Sleep -Seconds $interval
}
