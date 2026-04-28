# Full 5-policy benchmark (post-R1 reward).
# Runs sequentially: random -> reinforce -> vanilla_ac -> a2c -> ppo
# Each: budget=1000, 4 seeds (configs/default.yaml). ETA ~70 min.
$ErrorActionPreference = "Stop"
$py = "E:\--IDLE\python.exe"
$root = "f:\--CodeRepo\--CodeRepo\Research\___Mujoco\mujoco\mujoco\icse27\mujoco_rl_fuzz"
$benchLog = Join-Path $root "logs\bench_full.log"
Set-Location $root

function Stamp { (Get-Date).ToString("HH:mm:ss") }

"========== bench start  $(Get-Date)  ==========" | Tee-Object -FilePath $benchLog

# 1) random
"`n[$(Stamp)] >>> random_baseline budget=1000" | Tee-Object -FilePath $benchLog -Append
& $py -u -m src.experiments.random_baseline --config configs/default.yaml --budget 1000 2>&1 | Tee-Object -FilePath $benchLog -Append

# 2-5) RL algorithms
foreach ($algo in @("reinforce", "vanilla_ac", "a2c", "ppo")) {
    "`n[$(Stamp)] >>> rl_guided_fuzz --algorithm $algo budget=1000" | Tee-Object -FilePath $benchLog -Append
    & $py -u -m src.experiments.rl_guided_fuzz --config configs/default.yaml --algorithm $algo --budget 1000 2>&1 | Tee-Object -FilePath $benchLog -Append
}

"`n[$(Stamp)] >>> generating REPORT.md" | Tee-Object -FilePath $benchLog -Append
& $py -u "$root\tools\make_report.py" 2>&1 | Tee-Object -FilePath $benchLog -Append

"`n========== bench done   $(Get-Date)  ==========" | Tee-Object -FilePath $benchLog -Append
