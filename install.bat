@echo off
REM Bracket installer (Windows cmd.exe wrapper around install.ps1).
REM
REM Just runs the PowerShell installer with execution policy bypass for the
REM current process — no global policy change. Idempotent.
setlocal
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
