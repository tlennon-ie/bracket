@echo off
REM Bracket self-updater (Windows cmd.exe wrapper around update.ps1).
REM
REM Triggered by the React UI via POST /api/update/apply. The Python helper
REM in bracket/updater.py spawns this script detached from the dying API
REM process so the update can proceed even after uvicorn shuts down.
REM
REM Idempotent. Output is appended to runs\update.log.

setlocal
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
