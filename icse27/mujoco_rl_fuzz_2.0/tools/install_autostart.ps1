<#
.SYNOPSIS
    Register a Windows logon-time autostart so viz_gui2.py runs at user logon
    (port 9000, no console window). After install, every boot/logon you can
    just open http://127.0.0.1:9000/ — no manual start needed.

    NOTE: this targets viz_gui2.py (seeds 2.0, categorized & collapsible).
    The legacy viz_gui.py is no longer autostarted.

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
    pwsh tools\install_autostart.ps1 -Now
    pwsh tools\install_autostart.ps1 -PythonExe "E:\--IDLE\pythonw.exe" -Now
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
$TaskName = "MJFuzzVizGUI2"
$LegacyTaskName = "MJFuzzVizGUI"

# Project root = parent of this tools/ folder.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VizScript   = Join-Path $ProjectRoot "tools\viz_gui2.py"
$LogDir      = Join-Path $ProjectRoot ".cache"
$LogFile     = Join-Path $LogDir "viz_gui2.log"


if (-not (Test-Path $VizScript)) {
    throw "viz_gui2.py not found: $VizScript"
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
    if (Get-ItemProperty -Path $RunKey -Name $LegacyTaskName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty -Path $RunKey -Name $LegacyTaskName
        Write-Host "[ok] also removed legacy autostart entry '$LegacyTaskName'"
    }
    # Also kill any running instance
    Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "" } | ForEach-Object { $_.Kill() }
    return
}

# Always remove the legacy viz_gui.py autostart (user requested gui1 no longer auto-starts).
if (Get-ItemProperty -Path $RunKey -Name $LegacyTaskName -ErrorAction SilentlyContinue) {
    Remove-ItemProperty -Path $RunKey -Name $LegacyTaskName
    Write-Host "[ok] removed legacy viz_gui.py autostart entry '$LegacyTaskName'"
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
        Write-Host "[ok] viz_gui2 is listening at http://127.0.0.1:$Port/"
    } else {
        Write-Host "[warn] port $Port not responding yet — check for import errors:"
        Write-Host "       cd `"$ProjectRoot`"; python tools\viz_gui2.py"
    }
}

Write-Host ""
Write-Host "Manage:"
Write-Host "  Check  : Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' $TaskName"
Write-Host "  Remove : powershell -File $PSCommandPath -Uninstall"
