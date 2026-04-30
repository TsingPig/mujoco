' MJFuzzVizGUI silent autostart launcher.
' Spawns python.exe + viz_gui2.py with NO console window (intWindowStyle=0)
' and does NOT wait for it (bWaitOnReturn=False), so closing this script
' (which finishes immediately) does not kill the server.
'
' Registered via tools\install_autostart.ps1 in HKCU\...\Run as:
'   wscript.exe "<absolute path to this file>"
'
' Edit only via install_autostart.ps1 -- this file is regenerated.

Option Explicit

Dim shell, fso, scriptDir, projectRoot, vizScript, pyExe, port, cmdLine
Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")

scriptDir   = fso.GetParentFolderName(WScript.ScriptFullName)             ' ...\tools
projectRoot = fso.GetParentFolderName(scriptDir)                          ' ...\mujoco_rl_fuzz_2.0
vizScript   = fso.BuildPath(scriptDir, "viz_gui2.py")

' Read interpreter + port from sibling .ini (written by install_autostart.ps1).
Dim iniPath, ts, line, kv
iniPath = fso.BuildPath(scriptDir, "autostart_launcher.ini")
pyExe = ""
port  = "9000"
If fso.FileExists(iniPath) Then
    Set ts = fso.OpenTextFile(iniPath, 1, False)
    Do Until ts.AtEndOfStream
        line = Trim(ts.ReadLine)
        If Len(line) > 0 And Left(line, 1) <> ";" Then
            kv = Split(line, "=", 2)
            If UBound(kv) = 1 Then
                If LCase(Trim(kv(0))) = "python" Then pyExe = Trim(kv(1))
                If LCase(Trim(kv(0))) = "port"   Then port  = Trim(kv(1))
            End If
        End If
    Loop
    ts.Close
End If

If pyExe = "" Or Not fso.FileExists(pyExe) Then
    ' Fallback: try python on PATH (may pop a console; that's the user's choice).
    pyExe = "python.exe"
End If

cmdLine = """" & pyExe & """ """ & vizScript & """ --host 127.0.0.1 --port " & port & " --no-open"

' CurrentDirectory must be the project root so relative imports + seeds dir resolve.
shell.CurrentDirectory = projectRoot

' Run hidden (intWindowStyle=0), do NOT wait (bWaitOnReturn=False).
shell.Run cmdLine, 0, False
