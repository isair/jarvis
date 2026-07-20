' Launches Jarvis with no console window.
' pythonw.exe is the windowless Python host; the app lives in the system tray.
Option Explicit

Dim shell, fso, root, pyw, args
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = root & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pyw) Then
    MsgBox "Jarvis virtual environment not found at:" & vbCrLf & pyw, 16, "Jarvis"
    WScript.Quit 1
End If

' The app imports from src/, so PYTHONPATH must point there.
shell.Environment("PROCESS")("PYTHONPATH") = root & "\src"
shell.Environment("PROCESS")("PYTHONWARNINGS") = "ignore"

shell.CurrentDirectory = root
shell.Run """" & pyw & """ -m desktop_app", 0, False
