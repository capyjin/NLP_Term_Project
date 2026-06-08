#!/bin/bash
# ─────────────────────────────────────────────────────────────
# CNU Campus ChatBot — UI 빠른 실행 스크립트
#
# 사용:
#   bash run_ui.sh           # 데이터 갱신 + UI 실행
#   bash run_ui.sh --skip    # 데이터 갱신 건너뛰고 UI 바로 실행
#
# chatbot.sh 와의 차이:
#   chatbot.sh   : 전체 파이프라인 (크롤 + 추론 + UI)
#   run_ui.sh    : 데이터 갱신 TTL 체크 + UI 실행만
#                  (chat_output.json 생성 없음 → 데모용)
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  CNU Campus ChatBot — UI 실행"
echo "========================================"

# 1. 데이터 갱신 (--skip 없을 때만)
if [ "$1" != "--skip" ]; then
    echo ""
    echo "[1/2] 데이터 갱신 중..."
    python scripts/refresh_data.py || echo "  ⚠ 데이터 갱신 부분 실패 — 기존 파일로 계속"
fi

# 2. Gradio UI 실행
echo ""
echo "[2/2] UI 실행 중..."
echo "  접속: http://localhost:7860  (Colab: share URL 확인)"
echo "  종료: Ctrl+C"
echo ""
python src/chatbot_ui.py
