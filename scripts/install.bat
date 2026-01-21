@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

if not exist ".venv" (
  py -3.10 -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

echo.
echo Instalando FFmpeg local (sem passos manuais)...
set "TOOLS_DIR=%cd%\tools"
set "FFMPEG_DIR=%TOOLS_DIR%\ffmpeg"
set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
if not exist "%FFMPEG_BIN%\ffmpeg.exe" (
  if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
  if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$zip=Join-Path $env:TEMP 'ffmpeg-release-essentials.zip';" ^
    "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $zip;" ^
    "$dest=Join-Path $env:TEMP 'ffmpeg_extract_fx';" ^
    "if (Test-Path $dest) { Remove-Item -Recurse -Force $dest };" ^
    "Expand-Archive -Path $zip -DestinationPath $dest -Force;" ^
    "$bin=Get-ChildItem -Path $dest -Recurse -Filter ffmpeg.exe | Select-Object -First 1 | Split-Path -Parent;" ^
    "if (-not $bin) { throw 'Nao foi possivel localizar ffmpeg.exe no zip' };" ^
    "New-Item -ItemType Directory -Force -Path '%FFMPEG_BIN%' | Out-Null;" ^
    "Copy-Item -Path (Join-Path $bin '*') -Destination '%FFMPEG_BIN%' -Recurse -Force;" ^
    "Remove-Item -Recurse -Force $dest;" ^
    "Remove-Item -Force $zip;"
)

set "PATH=%FFMPEG_BIN%;%PATH%"

echo.
echo Instalando PyTorch com CUDA (tentando cu121, depois cu118, depois CPU)...
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
if errorlevel 1 (
  python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
)
if errorlevel 1 (
  python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
)

echo.
echo Instalando dependencias do projeto...
python -m pip install -r requirements.txt

echo.
echo Baixando modelos MiDaS/DPT para a pasta models (primeira execucao pode demorar)...
python -c "from app.depth import DepthConfig, DepthEstimator; import torch; cfg=DepthConfig(model_type='DPT_Large', device=torch.device('cpu'), fp16=False, depth_input_size=512); DepthEstimator(cfg).load(); print('OK')"

echo.
echo Instalacao concluida.
pause
