"""
데이터 갱신 스크립트 — 식단/셔틀/공지/장학 크롤링 + DB 재구축
────────────────────────────────────────────────────────────
사용:
  python scripts/refresh_data.py            # 전체 갱신
  python scripts/refresh_data.py --meal     # 식단만
  python scripts/refresh_data.py --shuttle  # 셔틀만
  python scripts/refresh_data.py --notice   # 공지/장학만
  python scripts/refresh_data.py --db       # 벡터 DB 재구축만
  python scripts/refresh_data.py --ui       # UI 바로 실행

chatbot.sh 와의 차이:
  - chatbot.sh : 전체 파이프라인 (크롤 → 추론 → UI)
  - refresh_data.py : 데이터만 선택적 갱신 (추론·UI 없음)
    → chroma_db 재구축 없이 raw 데이터만 갱신하고 싶을 때 사용
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── 색상 출력 헬퍼 ────────────────────────────────────────────────────────────
def _ok(msg):  print(f"  ✓ {msg}")
def _warn(msg): print(f"  ⚠ {msg}")
def _info(msg): print(f"  → {msg}")
def _sep(title=""):
    print(f"\n{'='*50}")
    if title: print(f"  {title}")
    if title: print(f"{'='*50}")


def run_crawler(script_path: Path, label: str) -> bool:
    """크롤러 실행. 실패해도 계속 진행 (fallback 파일 사용)."""
    _info(f"{label} 크롤링 중... ({script_path.name})")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if result.returncode == 0:
        _ok(f"{label} 갱신 완료")
        return True
    else:
        _warn(f"{label} 크롤링 실패 — 기존 파일 유지")
        if result.stderr:
            # 에러 첫 줄만 출력
            first_err = result.stderr.strip().splitlines()[-1][:120]
            print(f"     {first_err}")
        return False


def refresh_meal():
    """식단 크롤링 → data/raw/meal_menu.json 갱신."""
    _sep("식단 데이터 갱신")
    crawler = BASE_DIR / "src" / "crawling" / "meal_crawler.py"
    if not crawler.exists():
        _warn(f"meal_crawler.py 없음: {crawler}")
        return
    ok = run_crawler(crawler, "식단")
    if ok:
        raw = BASE_DIR / "data" / "raw" / "meal_menu.json"
        if raw.exists():
            import json
            data = json.loads(raw.read_text(encoding="utf-8"))
            _ok(f"meal_menu.json: {len(data)}건 저장됨")


def refresh_shuttle():
    """셔틀 크롤링 → data/raw/shuttle_*.json 갱신."""
    _sep("셔틀버스 데이터 갱신")
    crawler = BASE_DIR / "src" / "crawling" / "shuttle_crawler.py"
    if not crawler.exists():
        _warn(f"shuttle_crawler.py 없음: {crawler}")
        return
    run_crawler(crawler, "셔틀버스")


def refresh_academic_calendar():
    """학사일정 크롤링 → data/raw/academic_calendar.json 갱신 (TTL 24h 체크)."""
    _sep("학사일정 데이터 갱신")
    cal_json = BASE_DIR / "data" / "raw" / "academic_calendar.json"
    import time
    if cal_json.exists():
        age = time.time() - cal_json.stat().st_mtime
        if age < 24 * 3600:
            age_h = age / 3600
            _ok(f"학사일정 파일 유효 (갱신된 지 {age_h:.1f}시간) — 스킵")
            return
    crawler = BASE_DIR / "src" / "crawling" / "academic_calendar_crawler.py"
    if not crawler.exists():
        _warn(f"academic_calendar_crawler.py 없음: {crawler}")
        return
    run_crawler(crawler, "학사일정")


def refresh_notice():
    """공지/장학 크롤링 → chunks.json 갱신."""
    _sep("공지/장학 데이터 갱신")
    crawler = BASE_DIR / "src" / "crawling" / "cnu_crawler.py"
    if not crawler.exists():
        _warn(f"cnu_crawler.py 없음: {crawler}")
        _info("chunks.json의 기존 크롤 데이터를 유지합니다.")
        return
    run_crawler(crawler, "공지/장학")


def rebuild_db():
    """벡터 DB (chroma_db) 재구축."""
    _sep("벡터 DB 재구축")
    chroma = BASE_DIR / "chroma_db"

    # FAQ 재삽입
    inject = BASE_DIR / "scripts" / "inject_faq.py"
    if inject.exists():
        _info("FAQ 재삽입 중...")
        r = subprocess.run(
            [sys.executable, str(inject)],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        if r.returncode == 0:
            _ok("FAQ 재삽입 완료")
        else:
            _warn("FAQ 재삽입 실패 — 계속 진행")

    # chroma_db 재구축
    build = BASE_DIR / "src" / "vectordb" / "build_db.py"
    if not build.exists():
        _warn(f"build_db.py 없음: {build}")
        return
    _info("ChromaDB 임베딩 중... (~5~10분)")
    r = subprocess.run(
        [sys.executable, str(build)],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if r.returncode == 0:
        _ok("벡터 DB 재구축 완료")
    else:
        _warn("벡터 DB 재구축 실패")
        if r.stderr:
            print(f"     {r.stderr.strip().splitlines()[-1][:120]}")


def launch_ui():
    """Gradio UI 실행."""
    _sep("UI 실행")
    ui = BASE_DIR / "src" / "chatbot_ui.py"
    _info("Gradio UI 시작 중... (Ctrl+C로 종료)")
    subprocess.run([sys.executable, str(ui)], cwd=str(BASE_DIR))


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNU ChatBot 데이터 갱신 스크립트"
    )
    parser.add_argument("--meal",     action="store_true", help="식단 크롤링만")
    parser.add_argument("--shuttle",  action="store_true", help="셔틀 크롤링만")
    parser.add_argument("--notice",   action="store_true", help="공지/장학 크롤링만")
    parser.add_argument("--calendar", action="store_true", help="학사일정 크롤링만")
    parser.add_argument("--db",       action="store_true", help="벡터 DB 재구축만")
    parser.add_argument("--ui",       action="store_true", help="UI 바로 실행")
    args = parser.parse_args()

    # 플래그 없으면 전체 갱신 (UI 제외)
    run_all = not any([args.meal, args.shuttle, args.notice, args.calendar, args.db, args.ui])

    print("\n" + "=" * 50)
    print("  CNU ChatBot — 데이터 갱신")
    print("=" * 50)

    if run_all or args.meal:
        refresh_meal()

    if run_all or args.shuttle:
        refresh_shuttle()

    if run_all or args.notice:
        refresh_notice()

    if run_all or args.calendar:
        refresh_academic_calendar()

    if run_all or args.db:
        rebuild_db()

    if args.ui:
        launch_ui()

    if not args.ui:
        _sep("완료")
        _ok("데이터 갱신 완료. UI 실행: python src/chatbot_ui.py")


if __name__ == "__main__":
    main()
