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

if not exist "chroma_db" (
    echo.
    echo [1/3] 벡터 DB 구축 중 ^(처음 1회^)...
    python scripts\inject_faq.py
    python src\vectordb\build_db.py
)

echo.
echo [2/3] 챗봇 일괄 추론...
python src\chatbot_model.py --input data\test_chat.json --output outputs\chat_output.json

echo.
echo [3/3] UI 실행 중... ^(http://localhost:7860^)
python src\chatbot_ui.py

pause
