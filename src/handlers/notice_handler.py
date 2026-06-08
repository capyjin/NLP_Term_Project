"""
공지사항 핸들러
────────────────
역할: chunks.json의 학사공지·취업공지·행사안내 카테고리에서 최근 게시글 목록 반환.

처리 대상 질문:
  "최근 공지 알려줘", "학사 공지 보여줘", "취업 공지 뭐 올라왔어" 등
  → 목록/최신 조회 의도

처리 안 하는 질문:
  "장학금 공지" → ScholarshipHandler
  "수강신청 공지" → direct_handler / RAG

데이터 소스:
  data/processed/chunks.json
  카테고리: 학사공지(41개), 취업공지(30개), 행사안내(4개)
  ※ 일반공지(86개)는 홍보 보도자료 → 제외

정렬 기준:
  URL의 no= 번호 내림차순 (높은 번호 = 최신, ScholarshipHandler와 동일 방식)

날짜 표시:
  content에서 regex 추출 시도 (없으면 날짜 없이 표시)
  패턴: YYYY-MM-DD 또는 YYYY. M. D.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

# TTL: 6시간마다 갱신 (포털 업데이트 주기)
_NOTICE_TTL_SECONDS = 6 * 3600

# 포털 공지 게시판 URL
_PORTAL_URLS = {
    "학사공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0701&site_dvs_cd=kr&menu_dvs_cd=0701",
    "취업공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0704&site_dvs_cd=kr&menu_dvs_cd=0704",
    "행사안내": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0705&site_dvs_cd=kr&menu_dvs_cd=0705",
}
_PORTAL_DEFAULT = "https://plus.cnu.ac.kr/_prog/_board/?site_dvs_cd=kr"

# 카테고리 → 표시 레이블
_CATEGORY_LABEL = {
    "학사공지": "학사 공지사항",
    "취업공지": "취업 공지사항",
    "행사안내": "행사 안내",
}

# NoticeHandler가 처리하는 카테고리 (일반공지·장학공지 제외)
_VALID_CATEGORIES = frozenset({"학사공지", "취업공지", "행사안내"})


def _get_no(url: str) -> int:
    """URL에서 no= 번호 추출 (최신도 정렬용)."""
    m = re.search(r"no=(\d+)", url)
    return int(m.group(1)) if m else 0


def _extract_date(content: str) -> str:
    """
    content에서 날짜 추출.
    패턴 1: 2026-05-27
    패턴 2: 2026. 5. 27. (또는 2026. 5. 27)
    추출 실패 시 빈 문자열 반환.
    """
    # 패턴 1: ISO 형식
    m = re.search(r"(\d{4}-\d{2}-\d{2})", content)
    if m:
        return m.group(1)
    # 패턴 2: 한국 점 형식
    m = re.search(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", content)
    if m:
        y = m.group(1)
        mo = m.group(2).zfill(2)
        d = m.group(3).zfill(2)
        return f"{y}-{mo}-{d}"
    return ""


class NoticeHandler:
    """
    공지사항 목록 핸들러.
    answer(question) → (answer_text, source)
    source: "notice_handler"
    """

    def __init__(self, base_dir: Path):
        self._chunks_path = base_dir / "data" / "processed" / "chunks.json"
        self._cache: Optional[list] = None

    # ── TTL 캐시 ─────────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        """chunks.json이 없거나 6시간 초과 시 True."""
        if not self._chunks_path.exists():
            return True
        age = time.time() - self._chunks_path.stat().st_mtime
        return age > _NOTICE_TTL_SECONDS

    def _load_all(self) -> list[dict]:
        """chunks.json에서 공지 카테고리 청크 로드 + URL 중복 제거."""
        # 인메모리 캐시 우선
        if self._cache is not None:
            return self._cache

        if not self._chunks_path.exists():
            return []

        try:
            with open(self._chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception:
            return []

        # 유효 카테고리만 필터
        items = [c for c in chunks if c.get("category") in _VALID_CATEGORIES]

        # URL 기준 중복 제거 (같은 글이 여러 청크로 분할된 경우)
        seen: dict[str, dict] = {}
        for c in items:
            url = c.get("url", "")
            if url and url not in seen:
                seen[url] = c
            elif not url:
                # URL 없는 항목: title 기준 중복 제거
                key = c.get("title", "") or c.get("id", "")
                if key and key not in seen:
                    seen[key] = c

        self._cache = list(seen.values())
        return self._cache

    def _get_notices(self, categories: list[str]) -> list[dict]:
        """지정 카테고리만 필터 → no= 내림차순 정렬."""
        all_items = self._load_all()
        filtered = [c for c in all_items if c.get("category") in categories]
        filtered.sort(key=lambda c: _get_no(c.get("url", "")), reverse=True)
        return filtered

    # ── 포맷팅 ────────────────────────────────────────────────────────

    def _format_list(self, notices: list[dict], label: str, portal_url: str, top_n: int = 5) -> str:
        """최근 공지 목록 포매팅."""
        items = notices[:top_n]

        if not items:
            return (
                f"현재 저장된 {label} 데이터가 없습니다.\n"
                f"포털에서 직접 확인하세요: {portal_url}"
            )

        lines = [f"충남대학교 {label} (최근 {len(items)}건)\n"]
        for i, n in enumerate(items, 1):
            title = n.get("title", "") or "제목 없음"
            date  = _extract_date(n.get("content", ""))
            url   = n.get("url", "")

            date_str = f"[{date}] " if date else ""
            lines.append(f"{i}. {date_str}{title}")
            if url:
                lines.append(f"   → {url}")

        lines.append(f"\n전체 공지 확인: {portal_url}")
        return "\n".join(lines)

    # ── 메인 진입점 ───────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        """
        질문 분석 → 카테고리 결정 → 최근 공지 목록 반환.

        Returns: (answer_text, source)
        source: "notice_handler"
        """
        nq = question.replace(" ", "")

        # 카테고리 자동 감지
        if "취업" in nq:
            categories = ["취업공지"]
            label = _CATEGORY_LABEL["취업공지"]
            portal_url = _PORTAL_URLS["취업공지"]
        elif "학사" in nq:
            categories = ["학사공지"]
            label = _CATEGORY_LABEL["학사공지"]
            portal_url = _PORTAL_URLS["학사공지"]
        elif "행사" in nq:
            categories = ["행사안내"]
            label = _CATEGORY_LABEL["행사안내"]
            portal_url = _PORTAL_URLS["행사안내"]
        else:
            # 기본: 학사+취업+행사 통합 (최신순 혼합)
            categories = ["학사공지", "취업공지", "행사안내"]
            label = "공지사항 (학사·취업·행사)"
            portal_url = _PORTAL_URLS["학사공지"]

        notices = self._get_notices(categories)
        text = self._format_list(notices, label, portal_url)
        return text, "notice_handler"
