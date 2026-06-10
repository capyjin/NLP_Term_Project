"""
단과대/학과 소속 핸들러
━━━━━━━━━━━━━━━━━━━━━━
역할: 학과 → 단과대, 단과대 → 학과목록 조회

처리 대상:
  "컴퓨터인공지능학과 어디 단과대?" → 소속 단과대 반환
  "공과대학 학과 목록 보여줘"       → 소속 학과 목록 반환

매칭 전략:
  1. 정확 일치 (exact)
  2. 부분 포함 (substring)
  3. difflib 유사도 매칭 (cutoff=0.6)

데이터: data/raw/college_departments.json
"""

import json
import difflib
from pathlib import Path


class CollegeHandler:
    """단과대·학과 소속 조회 핸들러."""

    def __init__(self, base_dir: Path):
        data_path = base_dir / "data" / "raw" / "college_departments.json"
        try:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
            self._colleges: list[dict] = raw.get("colleges", [])
        except Exception:
            self._colleges = []

        # 역방향 인덱스: 학과명 → 단과대명
        self._dept_to_college: dict[str, str] = {}
        # 정방향 인덱스: 단과대명 → 학과목록
        self._college_to_depts: dict[str, list[str]] = {}

        for entry in self._colleges:
            college = entry["college"]
            depts   = entry["departments"]
            self._college_to_depts[college] = depts
            for dept in depts:
                self._dept_to_college[dept] = college

    # ── 퍼블릭 API ──────────────────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        """
        질문에 대한 (응답 텍스트, source) 반환.
        source 는 항상 "college_handler".
        """
        nq = question.replace(" ", "")

        # 단과대 → 학과목록 의도 우선 검사
        college_result = self._find_college(nq)
        if college_result:
            college_name, depts = college_result
            dept_list = "\n".join(f"  - {d}" for d in depts)
            ans = f"{college_name} 소속 학과/학부 목록입니다:\n{dept_list}"
            return ans, "college_handler"

        # 학과 → 단과대 조회
        dept_result = self._find_department(nq)
        if dept_result:
            matched_dept, college_name, is_fuzzy, original_query = dept_result
            if is_fuzzy and matched_dept != original_query:
                ans = (
                    f"입력하신 \"{original_query}\"와 가장 유사한 공식 명칭은 "
                    f"\"{matched_dept}\"입니다.\n"
                    f"{matched_dept}는 {college_name} 소속입니다."
                )
            else:
                ans = f"{matched_dept}는 {college_name} 소속입니다."
            return ans, "college_handler"

        # 매칭 실패 — 후보 제시
        ans = self._suggest(nq)
        return ans, "college_handler"

    # ── 내부 메서드 ─────────────────────────────────────────────────────────────

    def _find_department(self, nq: str) -> tuple[str, str, bool, str] | None:
        """
        학과명으로 단과대 찾기.
        Returns (matched_dept, college_name, is_fuzzy, original_query) or None.
        """
        all_depts = list(self._dept_to_college.keys())

        # 1단계: 정확 일치
        for dept in all_depts:
            if dept in nq:
                return dept, self._dept_to_college[dept], False, dept

        # 쿼리에서 학과/학부/전공으로 끝나는 단어 추출
        import re
        candidates = re.findall(r"[\w]+(?:학과|학부|전공|학교)", nq)
        if not candidates:
            # 단어가 없으면 nq 전체로 시도
            candidates = [nq]

        import re as _re
        for candidate in candidates:
            # 2단계: 접미사 제거한 stem으로 후보 좁힌 뒤 difflib
            stem = _re.sub(r"(학과|학부|전공|대학원)$", "", candidate)
            if stem and stem != candidate:
                stem_matches = [d for d in all_depts if stem in d]
                if stem_matches:
                    scored = difflib.get_close_matches(candidate, stem_matches, n=1, cutoff=0.5)
                    dept = scored[0] if scored else min(stem_matches, key=len)
                    is_exact = (dept == candidate)
                    return dept, self._dept_to_college[dept], not is_exact, candidate

            # 3단계: 전체 후보에 difflib (cutoff 높임)
            close = difflib.get_close_matches(candidate, all_depts, n=1, cutoff=0.7)
            if close:
                dept = close[0]
                return dept, self._dept_to_college[dept], True, candidate

            # 4단계: candidate가 dept를 포함하는 경우
            for dept in all_depts:
                if dept in candidate:
                    return dept, self._dept_to_college[dept], False, candidate

            # 5단계: candidate가 dept에 포함 (substring), 가장 짧은 것 우선
            partial_matches = [d for d in all_depts if candidate in d]
            if partial_matches:
                best = min(partial_matches, key=len)
                return best, self._dept_to_college[best], True, candidate

        return None

    def _find_college(self, nq: str) -> tuple[str, list[str]] | None:
        """
        단과대명으로 학과목록 찾기.
        Returns (college_name, depts) or None.
        """
        all_colleges = list(self._college_to_depts.keys())

        # 1단계: 정확 일치
        for college in all_colleges:
            if college in nq:
                return college, self._college_to_depts[college]

        # 2단계: 부분 포함
        for college in all_colleges:
            # 단과대 이름 핵심 부분으로 매칭 (예: "공과" in "공과대학")
            core = college.replace("대학", "").replace("학부", "")
            if core and core in nq:
                return college, self._college_to_depts[college]

        # 3단계: difflib
        close = difflib.get_close_matches(nq[:10], all_colleges, n=1, cutoff=0.5)
        if close:
            college = close[0]
            return college, self._college_to_depts[college]

        return None

    def _suggest(self, nq: str) -> str:
        """매칭 실패 시 후보 학과명 제시."""
        all_depts = list(self._dept_to_college.keys())
        close = difflib.get_close_matches(nq, all_depts, n=3, cutoff=0.4)
        if close:
            suggestions = ", ".join(f"\"{d}\"" for d in close)
            return (
                f"해당 학과/단과대를 찾지 못했습니다.\n"
                f"혹시 다음 중 하나를 찾으시나요? {suggestions}"
            )
        return (
            "해당 학과 또는 단과대 정보를 찾지 못했습니다.\n"
            "정확한 학과명이나 단과대명을 입력해 주세요.\n"
            "예시: \"컴퓨터인공지능학과 어디 단과대?\", \"공과대학 학과 목록\""
        )
