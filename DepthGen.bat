@echo off
title DepthGen
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo   ==============================
echo    DepthGen - plaskorzezby 3D
echo   ==============================
echo.

rem --- srodowisko ---
if not exist ".venv\Scripts\python.exe" (
  echo   [!] Brak srodowiska Python.
  echo.
  choice /c TN /n /m "   Zainstalowac teraz? Potrwa kilka minut i pobierze ok. 3 GB [T/N]: "
  if errorlevel 2 exit /b 1
  call setup.bat
  if not exist ".venv\Scripts\python.exe" (
    echo   [!] Instalacja nie powiodla sie.
    pause & exit /b 1
  )
)

set PY=.venv\Scripts\python.exe

rem --- wolny port od 8077 w gore ---
set PORT=
for /l %%p in (8077,1,8097) do (
  if not defined PORT (
    netstat -ano | findstr /r /c:":%%p .*LISTENING" >nul 2>&1
    if errorlevel 1 set PORT=%%p
  )
)
if not defined PORT (
  echo   [!] Brak wolnego portu w zakresie 8077-8097.
  pause & exit /b 1
)

echo   Adres:   http://127.0.0.1:%PORT%
"%PY%" -c "import torch;print('   Sprzet:  '+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (bez CUDA - bedzie wolniej)'))" 2>nul
echo   Zamkniecie: to okno lub Ctrl+C
echo.

rem --- przegladarka po starcie serwera ---
start "" /b cmd /c "ping -n 5 127.0.0.1 >nul & start "" http://127.0.0.1:%PORT%"

"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% --log-level warning

echo.
echo   Serwer zatrzymany.
pause
