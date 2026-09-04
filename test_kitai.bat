@echo off
setlocal
cd /d "%~dp0"
"C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe" scripts\talk_to_kitai.py --quick-test --max-tokens 64
pause
