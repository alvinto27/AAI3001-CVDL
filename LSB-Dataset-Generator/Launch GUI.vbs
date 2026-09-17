Option Explicit
Dim fso, shell, root, runtimeFile, pythonw, reader
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonw) Then
    runtimeFile = root & "\.runtime-path.txt"
    If fso.FileExists(runtimeFile) Then
        Set reader = fso.OpenTextFile(runtimeFile, 1)
        pythonw = Trim(reader.ReadLine)
        reader.Close
    End If
End If
If Not fso.FileExists(pythonw) Then
    MsgBox "Python is not configured. See README.md for setup, then open this launcher again.", 48, "LSB Dataset Generator"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & root & "\launch.pyw" & Chr(34), 1, False
