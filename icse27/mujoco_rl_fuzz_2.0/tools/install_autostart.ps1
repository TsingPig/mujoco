<#
.SYNOPSIS
    Register a Windows Scheduled Task so viz_gui.py runs at user logon
    (port 9000, no console window). After install, every boot/logon you can
    just open http://127.0.0.1:9000/ — no manual start needed.

.PARAMETER Port
    HTTP port (default 9000).

.PARAMETER PythonExe
    Path to pythonw.exe (preferred — no flashing console). If empty, auto-resolved
    from the python.exe on PATH.

.PARAMETER Uninstall
    Remove the scheduled task instead of installing it.

.PARAMETER Now
    Also kick off the task immediately after installing.

.EXAMPLES
    # Install and start now
    pwsh tools\install_autostart.ps1 -Now

    # Use a specific interpreter
    pwsh tools\install_autostart.ps1 -PythonExe "E:\--IDLE\pythonw.exe" -Now

    # Remove
    pwsh tools\install_autostart.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [int]$Port = 9000,
    [string]$PythonExe = "",
    [switch]$Uninstall,
    [switch]$Now
)

$ErrorActionPreference = "Stop"
$TaskName = "MJFuzzVizGUI"

# Project root = parent of this tools/ folder.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VizScript   = Join-Path $ProjectRoot "tools\viz_gui.py"
$LogDir      = Join-Path $ProjectRoot ".cache"
$LogFile     = Join-Path $LogDir "viz_gui.log"


if (-not (Test-Path $VizScript)) {
    throw "viz_gui.py not found: $VizScript"
}

# Resolve pythonw.exe (preferred to suppress console window).
if (-not $PythonExe) {
    $py = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $py) {
        throw "python.exe not on PATH; pass -PythonExe explicitly."
    }
    $candidate = Join-Path (Split-Path $py.Path) "pythonw.exe"
    if (Test-Path $candidate) {
        $PythonExe = $candidate
    } else {
        Write-Warning "pythonw.exe not found next to $($py.Path); falling back to python.exe (will keep a console window)."
        $PythonExe = $py.Path
    }
}
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$CmdLine = "`"$PythonExe`" `"$VizScript`" --host 127.0.0.1 --port $Port --no-open"

# Use HKCU Run registry key — no admin rights required, current user only.
# Task Scheduler alternatives (Register-ScheduledTask) require elevation on PS 5.1.
$RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"

if ($Uninstall) {
    if (Get-ItemProperty -Path $RunKey -Name $TaskName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty -Path $RunKey -Name $TaskName
        Write-Host "[ok] removed autostart entry '$TaskName' from registry Run key"
    } else {
        Write-Host "[noop] '$TaskName' not found in registry Run key"
    }
    # Also kill any running instance
    Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "" } | ForEach-Object { $_.Kill() }
    return
}

# Write (or overwrite) the Run entry
Set-ItemProperty -Path $RunKey -Name $TaskName -Value $CmdLine
Write-Host "[ok] registered autostart: HKCU\...\Run\$TaskName"
Write-Host "     interpreter : $PythonExe"
Write-Host "     script      : $VizScript"
Write-Host "     port        : $Port"
Write-Host "     workdir     : $ProjectRoot"
Write-Host "     trigger     : at user logon (registry Run key)"
Write-Host ""

if ($Now) {
    # Start the server in background now (don't wait)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $PythonExe
    $psi.Arguments = "`"$VizScript`" --host 127.0.0.1 --port $Port --no-open"
    $psi.WorkingDirectory = $ProjectRoot
    $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $psi.CreateNoWindow = $true
    $psi.UseShellExecute = $false
    [System.Diagnostics.Process]::Start($psi) | Out-Null

    Write-Host "[..] viz_gui.py launched in background, waiting for port $Port ..."
    $deadline = (Get-Date).AddSeconds(8)
    $ok = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 600
        try {
            $r = [System.Net.Sockets.TcpClient]::new("127.0.0.1", $Port)
            $r.Close(); $ok = $true; break
        } catch {}
    }
    if ($ok) {
        Write-Host "[ok] viz_gui is listening at http://127.0.0.1:$Port/"
    } else {
        Write-Host "[warn] port $Port not responding yet — check for import errors:"
        Write-Host "       cd `"$ProjectRoot`"; python tools\viz_gui.py"
    }
}

Write-Host ""
Write-Host "Manage:"
Write-Host "  Check  : Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' $TaskName"
Write-Host "  Remove : powershell -File $PSCommandPath -Uninstall"
