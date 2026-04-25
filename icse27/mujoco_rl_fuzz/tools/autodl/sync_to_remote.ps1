<#
.SYNOPSIS
  Sync this repo to an AutoDL instance via rsync over SSH.

.DESCRIPTION
  AutoDL gives you a host like  region-X.seetacloud.com  port 12345
  user `root`, password `xxxxxx`. Set them in environment variables OR pass
  -Host -Port -User. Requires `rsync` on PATH (use Git-Bash / MSYS rsync.exe
  on Windows, or run this script from WSL).

.EXAMPLE
  $env:AUTODL_HOST="connect.westa.seetacloud.com"
  $env:AUTODL_PORT="12345"
  ./tools/autodl/sync_to_remote.ps1
#>
param(
  [string]$RemoteHost = $env:AUTODL_HOST,
  [int]   $Port = [int]($env:AUTODL_PORT ?? 22),
  [string]$User = ($env:AUTODL_USER ?? "root"),
  [string]$RemotePath = "/root/mujoco_rl_fuzz",
  [switch]$Pull   # pull artifacts back instead of pushing source
)

if (-not $RemoteHost) { throw "Set -RemoteHost or env:AUTODL_HOST" }

$LocalRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repo root
$Excludes  = @(".cache", "__pycache__", "*.pyc", "outputs/crashes",
               "outputs/warnings", "outputs/interesting", "logs/*.jsonl",
               "checkpoints", "data", ".git", ".venv")
$ExArgs = $Excludes | ForEach-Object { "--exclude=$_" }

if ($Pull) {
  Write-Host "[pull] $User@${RemoteHost}:${RemotePath}/{checkpoints,logs,outputs} -> $LocalRoot"
  rsync -avz -e "ssh -p $Port" `
        "${User}@${RemoteHost}:${RemotePath}/checkpoints" "$LocalRoot/"
  rsync -avz -e "ssh -p $Port" `
        "${User}@${RemoteHost}:${RemotePath}/logs"        "$LocalRoot/"
  rsync -avz -e "ssh -p $Port" `
        "${User}@${RemoteHost}:${RemotePath}/outputs"     "$LocalRoot/"
} else {
  Write-Host "[push] $LocalRoot -> $User@${RemoteHost}:${RemotePath}"
  rsync -avz -e "ssh -p $Port" $ExArgs `
        "$LocalRoot/" "${User}@${RemoteHost}:${RemotePath}/"
}
