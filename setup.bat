@echo off
setlocal
cd /d "%~dp0"

echo [DepthGen] Tworzenie srodowiska Python...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv || (echo Brak Pythona 3.10+ w PATH & pause & exit /b 1)
)

".venv\Scripts\python.exe" -m pip install --upgrade pip

echo.
echo [DepthGen] PyTorch z obsluga CUDA (~2.5 GB, chwile potrwa)...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo [DepthGen] CUDA sie nie powiodla - instaluje wersje CPU
  ".venv\Scripts\python.exe" -m pip install torch torchvision
)

echo.
echo [DepthGen] Pozostale biblioteki...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo [DepthGen] Gotowe. Uruchom run.bat
pause
