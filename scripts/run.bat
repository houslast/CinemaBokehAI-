@echo off
setlocal

cd /d "%~dp0\.."

if not exist ".venv\Scripts\activate.bat" (
  echo Venv nao encontrado. Rode scripts\install.bat primeiro.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat

set "FFMPEG_BIN=%cd%\tools\ffmpeg\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" (
  set "PATH=%FFMPEG_BIN%;%PATH%"
)

set "PYTHONPATH=%cd%;%PYTHONPATH%"

streamlit run app\main.py
