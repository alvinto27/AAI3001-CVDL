Option Explicit
' Start the LSB Dataset Generator GUI with the first usable Python found.
' No particular Python version is required. Python 3.11 or newer with Tcl/Tk
' support is enough. Resolution order: project .venv, the path recorded in
' .runtime-path.txt, the Windows Python launcher, an installed Python, then
' pythonw.exe on PATH.

Dim fso, shell, root, pythonw, extraArgs, attempts, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
extraArgs = ""
attempts = ""

pythonw = FindPythonw()

If pythonw = "" Then
    MsgBox "Python could not be found, so the generator cannot start." & vbCrLf & vbCrLf & _
        "Locations searched:" & vbCrLf & attempts & vbCrLf & _
        "Install Python 3.11 or newer with Tcl/Tk support and try again, or follow " & _
        "the setup commands in README.md to create a virtual environment.", _
        48, "LSB Dataset Generator"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
command = Chr(34) & pythonw & Chr(34)
If Len(extraArgs) > 0 Then
    command = command & " " & extraArgs
End If
command = command & " " & Chr(34) & root & "\launch.pyw" & Chr(34)
shell.Run command, 1, False


Function FindPythonw()
    Dim path, launcher, folder, candidate, best, bestName, runtimeFile, reader, entry
    FindPythonw = ""

    ' 1. Project virtual environment created by the README setup.
    path = root & "\.venv\Scripts\pythonw.exe"
    attempts = attempts & "  " & path & "  (project virtual environment)" & vbCrLf
    If fso.FileExists(path) Then
        FindPythonw = path
        Exit Function
    End If

    ' 2. Path recorded by an earlier setup on this computer.
    runtimeFile = root & "\.runtime-path.txt"
    If fso.FileExists(runtimeFile) Then
        Set reader = fso.OpenTextFile(runtimeFile, 1)
        If Not reader.AtEndOfStream Then
            path = Trim(reader.ReadLine)
        End If
        reader.Close
        attempts = attempts & "  " & path & "  (from .runtime-path.txt)" & vbCrLf
        If Len(path) > 0 Then
            If fso.FileExists(path) Then
                FindPythonw = path
                Exit Function
            End If
        End If
    End If

    ' 3. Windows Python launcher. It selects the newest installed Python 3 and
    '    runs a .pyw script without a console window.
    launcher = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Launcher\"
    path = launcher & "pyw.exe"
    attempts = attempts & "  " & path & "  (Python launcher)" & vbCrLf
    If fso.FileExists(path) Then
        FindPythonw = path
        extraArgs = "-3"
        Exit Function
    End If

    ' 4. Newest full Python installation in the usual locations.
    best = ""
    bestName = ""
    For Each folder In Array(shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python", _
                             shell.ExpandEnvironmentStrings("%ProgramFiles") & "\Python")
        If fso.FolderExists(folder) Then
            For Each candidate In fso.GetFolder(folder).SubFolders
                path = candidate.Path & "\pythonw.exe"
                If fso.FileExists(path) Then
                    If best = "" Then
                        best = path
                        bestName = candidate.Name
                    ElseIf StrComp(candidate.Name, bestName, vbTextCompare) > 0 Then
                        best = path
                        bestName = candidate.Name
                    End If
                End If
            Next
        End If
    Next
    If best <> "" Then
        attempts = attempts & "  " & best & "  (installed Python)" & vbCrLf
        FindPythonw = best
        Exit Function
    End If

    ' 5. pythonw.exe anywhere on PATH.
    For Each entry In Split(shell.ExpandEnvironmentStrings("%PATH%"), ";")
        folder = Trim(entry)
        If Len(folder) > 0 Then
            If Right(folder, 1) = "\" Then
                path = folder & "pythonw.exe"
            Else
                path = folder & "\pythonw.exe"
            End If
            If fso.FileExists(path) Then
                attempts = attempts & "  " & path & "  (pythonw.exe on PATH)" & vbCrLf
                FindPythonw = path
                Exit Function
            End If
        End If
    Next
    attempts = attempts & "  pythonw.exe on PATH" & vbCrLf
End Function
