"""
장학공지 핸들러
────────────────
역할: chunks.json의 장학공지 크롤링 데이터에서 실제 장학 관련 게시글 목록 반환.

처리 대상 질문:
  "최근 장학금 리스트 보여줘", "장학공지 알려줘", "장학금 뭐 있어?" 등
  → 리스트/목록 의도 질문

처리 안 하는 질문:
  "장학금 신청 방법", "얼마 받을 수 있어" 등
  → RAG/FAQ로 처리 (설명형 질문)

데이터 소스:
  data/processed/chunks.json → category="장학공지", source_type="crawl"
  URL의 no= 번호 기준 내림차순 정렬 (높은 번호 = 최신)
  날짜 필드 없음 → content에서 regex 추출 시도

주의:
  - 장학공지 게시판(code=sub07_0702)에는 계절학기·취업·멘토 등 비장학 글도 올라옴
  - _SCHOLAR_TITLE_KW 화이트리스트로 실제 장학 관련 게시글만 선별
  - 교수/포스트닥 대상 항목은 추가 블랙리스트 필터링
  - 허위 장학금 생성 금지 -- 반드시 실제 크롤링 데이터만 사용
"""

import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

# TTL: 장학공지는 6시간마다 갱신 (포털 공지 업데이트 주기)
_SCHOLAR_TTL_SECONDS = 6 * 3600

PORTAL_URL     = "https://plus.cnu.ac.kr"
SCHOLAR_BOARD  = "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0702&site_dvs_cd=kr&menu_dvs_cd=0702"

# ── 장학 관련 제목 화이트리스트 ──────────────────────────────────────────────
# 장학공지 게시판에 비장학 글이 섞이므로, 제목에 아래 키워드가 있는 것만 장학공지로 인정
_SCHOLAR_TITLE_KW = frozenset({
    "장학",       # 장학금, 장학생, 장학프로그램 등 전부 포함
    "드림클래스",  # 삼성드림클래스
    "파란사다리",  # 교육부 파란사다리
    "한국장학",   # 한국장학재단
    "지원금",     # 각종 학생 지원금
    "학비지원",
    "등록금지원",
})

# ── 비학생 대상 블랙리스트 (화이트리스트 통과 후 추가 제외) ────────────────────
# "장학" 키워드가 있어도 교수·대학원 전용이면 제외
_PROF_FILTER_KW = frozenset({
    "풀브라이트", "포스트닥", "신약전문대학원", "연구년",
    "교원", "교수", "교직원", "대학원신입생", "보충강의",
    "연구비지원", "외국인교원",
})


def _get_no(url: str) -> int:
    """URL에서 no= 번호 추출 (최신도 추정용)."""
    m = re.search(r"no=(\d+)", url)
    return int(m.group(1)) if m else 0


def _extract_date(content: str) -> str:
    """
    content에서 날짜 추출.
    패턴 1: 등록일YYYY-MM-DD
    패턴 2: 접수기간YYYY-MM-DD
    패턴 3: YYYY-MM-DD (오늘 이전 날짜만)
    패턴 4: YYYY. M. D.
    추출 실패 시 빈 문자열 반환.
    """
    m = re.search(r"등록일\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        return m.group(1)
    m = re.search(r"접수기간\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        return m.group(1)
    today_str = date.today().isoformat()
    for m in re.finditer(r"(\d{4}-\d{2}-\d{2})", content):
        d = m.group(1)
        if d <= today_str:
            return d
    m = re.search(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", content)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return ""


class ScholarshipHandler:
    """
    장학공지 리스트 핸들러.
    answer(question) → (answer_text, source)
    source: "scholarship_handler" | "scholarship_no_data"
    """

    def __init__(self, base_dir: Path):
        self._chunks_path = base_dir / "data" / "processed" / "chunks.json"
        self._crawler     = base_dir / "src" / "crawling" / "cnu_crawler.py"
        self._cache: Optional[list] = None

    # ── TTL 캐시 ─────────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        """chunks.json이 없거나 6시간 초과 시 True."""
        if not self._chunks_path.exists():
            return True
        age = time.time() - self._chunks_path.stat().st_mtime
        return age > _SCHOLAR_TTL_SECONDS

    def _try_refresh(self) -> None:
        """cnu_crawler 실행으로 chunks.json 갱신 시도. 실패 시 기존 유지."""
        if not self._crawler.exists():
            print("[TTL] cnu_crawler.py 없음 -- 기존 chunks.json 사용")
            return
        print("[TTL] chunks.json stale -- refreshing scholarship data...")
        result = subprocess.run(
            [sys.executable, str(self._crawler)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("[TTL] scholarship refresh success")
            self._cache = None  # 인메모리 캐시 초기화
        else:
            err = (result.stderr or "").strip().splitlines()
            print("[TTL] scholarship refresh failed -- using cached file")
            if err:
                print(f"      {err[-1][:100]}")

    def _load_notices(self) -> list[dict]:
        """chunks.json → 장학공지 크롤링 청크 → URL 중복 제거 → 최신순 정렬."""
        # 인메모리 캐시 우선
        if self._cache is not None:
            print("[TTL] scholarship using cached (in-memory)")
            return self._cache

        # TTL 체크: stale이면 크롤러 실행
        if self._is_stale():
            self._try_refresh()
        else:
            print("[TTL] chunks.json fresh -- using cached file")

        if not self._chunks_path.exists():
            return []

        try:
            with open(self._chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception:
            return []

        # 장학공지 게시판 청크만 (FAQ 제외)
        scholar = [
            c for c in chunks
            if c.get("category") == "장학공지"
            and c.get("source_type", "") != "faq_manual"
        ]

        # URL 기준 중복 제거 (같은 URL = 같은 게시글)
        seen: dict[str, dict] = {}
        for c in scholar:
            url = c.get("url", "")
            if url and url not in seen:
                seen[url] = c

        # ① 장학 관련 제목 화이트리스트 — 비장학 게시글 제외
        # 장학공지 게시판에는 계절학기·취업·멘토 등 비장학 글도 올라오므로
        # 제목에 장학 관련 키워드가 있는 것만 인정
        notices = [
            n for n in seen.values()
            if any(kw in n.get("title", "") for kw in _SCHOLAR_TITLE_KW)
        ]

        # ② 교수/대학원 전용 블랙리스트 추가 제외
        notices = [
            n for n in notices
            if not any(kw in n.get("title", "") for kw in _PROF_FILTER_KW)
        ]

        # no= 번호 기준 내림차순 (최신 먼저)
        notices.sort(key=lambda c: _get_no(c.get("url", "")), reverse=True)

        self._cache = notices
        return notices

    def _format_list(self, notices: list[dict], top_n: int = 5) -> str:
        """최근 장학/지원 공지 목록 포매팅 (날짜 포함)."""
        items = notices[:top_n]
        total = len(items)

        # 헤더: 5건 미만이면 자연스럽게 안내
        if total < 5:
            header = (
                f"충남대학교 최근 장학/지원 관련 공지 {total}건\n"
                f"(현재 저장된 장학/지원 관련 공지는 {total}건입니다.)\n"
            )
        else:
            header = f"충남대학교 최근 장학/지원 관련 공지 {total}건\n"

        lines = [header]

        for i, n in enumerate(items, 1):
            title    = n.get("title", "제목 없음")
            url      = n.get("url", "")
            d        = _extract_date(n.get("content", ""))
            date_str = f"[{d}] " if d else "[날짜 미확인] "
            lines.append(f"{i}. {date_str}{title}")
            if url:
                lines.append(f"   → {url}")

        lines.append(f"\n전체 장학공지 확인: {SCHOLAR_BOARD}")
        lines.append(
            "※ 장학공지 게시판에는 장학금 외에도 학생지원·멘토·파란사다리 등 "
            "지원성 공지가 포함될 수 있습니다."
        )
        return "\n".join(lines)

    def answer(self, question: str) -> tuple[str, str]:
        """
        Returns: (answer_text, source)
        source: "scholarship_handler" | "scholarship_no_data"
        """
        notices = self._load_notices()

        if not notices:
            return (
                "현재 저장된 장학공지 데이터가 없습니다.\n"
                f"충남대 포털 장학공지에서 직접 확인하세요:\n{SCHOLAR_BOARD}",
                "scholarship_no_data",
            )

        text = self._format_list(notices, top_n=10)
        return text, "scholarship_handler"
