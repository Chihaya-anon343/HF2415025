@echo off
setlocal
for %%I in ("%~dp0.") do set "STERETRACKER_RUNTIME_ROOT=%%~fI"
set "STERETRACKER_RUNTIME_CONTAINER=st_runtime_standalone_20261117"
if "%~1"=="" (
  echo Usage: run_docker.cmd mono^|mono-fused^|stereo^|stereo-disparity [--max-frames N]
  exit /b 2
)
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker is not available. Open Docker Desktop and retry.
  exit /b 1
)
docker image inspect stereotracker_env:latest >nul 2>&1
if errorlevel 1 (
  echo Required image stereotracker_env:latest was not found.
  exit /b 1
)
docker container inspect %STERETRACKER_RUNTIME_CONTAINER% >nul 2>&1
if errorlevel 1 (
  docker run -d --name %STERETRACKER_RUNTIME_CONTAINER% --mount "type=bind,source=%STERETRACKER_RUNTIME_ROOT%,target=/workspace/runtime" -w /workspace/runtime stereotracker_env:latest sleep infinity
  if errorlevel 1 exit /b 1
)
powershell -NoProfile -Command "$c=(docker inspect $env:STERETRACKER_RUNTIME_CONTAINER | ConvertFrom-Json)[0]; $m=@($c.Mounts | Where-Object {$_.Destination -eq '/workspace/runtime'}); if($m.Count -ne 1 -or $m[0].Source.Replace('/','').TrimEnd('') -ine $env:STERETRACKER_RUNTIME_ROOT.TrimEnd('')) { Write-Error 'Container mount does not match this project. No experiment started.'; exit 1 }"
if errorlevel 1 exit /b 1
docker start %STERETRACKER_RUNTIME_CONTAINER% >nul
if errorlevel 1 exit /b 1
docker exec --workdir /workspace/runtime %STERETRACKER_RUNTIME_CONTAINER% python3 -u run.py %*
exit /b %errorlevel%
