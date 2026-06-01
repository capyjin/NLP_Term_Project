#!/bin/bash
# ─────────────────────────────────────────────────────────────
# CNU Campus ChatBot — 평가 실행 스크립트
# 사용: bash chatbot.sh
#
# 1. outputs/ 폴더 생성
# 2. chroma_db 없으면 벡터 DB 자동 구축 (처음 1회만)
# 3. data/test_chat.json → outputs/chat_output.json 생성
# 4. Gradio UI 실행 (http://localhost:7860)
# ─────────────────────────────────────────────────────────────

set -e

# 프로젝트 루트를 스크립트 위치로 고정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  CNU Campus ChatBot"
echo "========================================"

# 1. outputs/ 폴더 보장
mkdir -p outputs

# 2. chroma_db 없으면 구축 (FAQ 삽입 → 임베딩 → DB 저장)
if [ ! -d "chroma_db" ]; then
    echo ""
    echo "[1/3] 벡터 DB 구축 중 (처음 1회, ~10분 소요)..."
    python scripts/inject_faq.py
    python src/vectordb/build_db.py
    echo "벡터 DB 구축 완료"
fi

# 3. 식단/셔틀버스 데이터 크롤링 (실패해도 계속 진행 — 수동 파일 또는 known_data 사용)
echo ""
echo "[2/4] 식단/셔틀버스 데이터 업데이트 중..."
python src/crawling/meal_crawler.py    || echo "  ⚠ 식단 크롤링 실패 — data/raw/meal_menu.json 수동 파일 사용"
python src/crawling/shuttle_crawler.py || echo "  ⚠ 셔틀버스 크롤링 실패 — known_data 사용"

# 4. test_chat.json → chat_output.json
#    식단/셔틀 질문 → 핸들러 처리 / 그 외 → RAGPipeline
echo ""
echo "[3/4] 챗봇 일괄 추론: data/test_chat.json → outputs/chat_output.json"
python src/chatbot_model.py \
    --input  data/test_chat.json \
    --output outputs/chat_output.json

# [Optional] test_realtime.json 있으면 실시간 정보 처리
if [ -f "data/test_realtime.json" ]; then
    echo ""
    echo "[Optional] 실시간 정보 처리: data/test_realtime.json → outputs/realtime_output.json"
    python src/realtime_model.py \
        --input  data/test_realtime.json \
        --output outputs/realtime_output.json
fi

# 5. Gradio UI 실행
echo ""
echo "[4/4] UI 실행 중... (http://localhost:7860)"
echo "종료하려면 Ctrl+C"
python src/chatbot_ui.py
