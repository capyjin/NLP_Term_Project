@echo off
REM ─────────────────────────────────────────────────────────────
REM CNU Campus ChatBot — Windows 실행 스크립트
REM 사용: chatbot.bat (또는 더블클릭)
REM ─────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo ========================================
echo   CNU Campus ChatBot
echo ========================================

mkdir outputs 2>nul

python scripts\inject_faq.py
if errorlevel 1 exit /b 1
python scripts\check_vector_db.py
if errorlevel 1 (
    echo.
    echo [1/3] 벡터 DB 재구축 중...
    python src\vectordb\build_db.py --fresh
    if errorlevel 1 exit /b 1
)

echo.
echo [2/3] 챗봇 일괄 추론...
python src\chatbot_model.py --input data\test_chat.json --output outputs\chat_output.json

echo.
echo [3/3] UI 실행 중... ^(http://localhost:7860^)
python src\chatbot_ui.py

pause
