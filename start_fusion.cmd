@echo off
call "%~dp0run_docker.cmd" mono-fused %*
set "RUN_EXIT=%errorlevel%"
echo.
echo Exit code: %RUN_EXIT%
pause
exit /b %RUN_EXIT%
