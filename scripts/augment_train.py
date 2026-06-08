"""
train.json 데이터 증강 스크립트
────────────────────────────────
목표: 각 label 40~45개 (train 총 200~225개)
현재: {0:30, 1:30, 2:30, 3:28, 4:28} = 146개

생성 기준:
  - 실제 충남대 학생 질문 스타일
  - 존댓말 / 반말 / 오타 / 짧은 질문 / 긴 질문 / 애매한 표현
  - label 충돌 방지: 식단(3)/셔틀(4) 키워드가 없는 라벨 0~2만 추가

실행:
  python scripts/augment_train.py          # 증강 후 train.json 덮어쓰기
  python scripts/augment_train.py --dry    # 변경 없이 미리보기만
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE / "data" / "train.json"

# ── 증강 데이터 ───────────────────────────────────────────────────────────────
# 각 label당 추가 질문. 기존 데이터와 중복 최소화.

_AUGMENT = {
    # 0: 졸업요건
    0: [
        "졸업하려면 총 몇 학점이야?",
        "전공 필수 몇 개 들어야 해?",
        "교양 학점 얼마나 필요해요?",
        "컴퓨터공학과 졸업 요건 알려줘",
        "졸업인증제가 뭐예요?",
        "영어 졸업인증 어떻게 해요?",
        "조기졸업 조건이 뭐예요?",
        "편입생 졸업학점 달라요?",
        "복수전공하면 졸업학점 더 필요해요?",
        "학생편람 어디서 봐요?",
        "졸업 요건 충족했는지 어떻게 확인해요?",
        "이수 학점 얼마나 남았는지 어떻게 알아요?",
        "졸업예정자 확인 어디서 해요?",
        "전공 학점 부족하면 졸업 못 해요?",
        "졸업 요건이 학번마다 달라요?",
    ],

    # 1: 학교공지사항
    1: [
        "최근 학교 공지 보여줘",
        "학사 공지사항 어디서 봐요?",
        "충남대 포털 공지 확인 방법 알려줘",
        "학교 행사 일정 어디서 확인해요?",
        "취업 공지 어디서 봐요?",
        "최근 학교 뉴스 뭐 있어요?",
        "학교 홈페이지 공지 어디 있어요?",
        "일반 공지 어디 있어요?",
        "학교 소식 알려줘",
        "입학처 공지 어떻게 봐요?",
        "학생처 공지사항 확인 방법 알려줘",
        "장학공지 어디서 확인해요?",
        "학교 행사나 이벤트 공지 어디서 봐요?",
        "공지사항 알림 받는 방법 있어요?",
        "충남대 새로운 소식 어떻게 확인해요?",
    ],

    # 2: 학사일정
    2: [
        "이번 학기 종강 언제예요?",
        "개강이 언제야?",
        "중간고사 일정 알려줘",
        "기말고사 언제야?",
        "방학이 언제예요?",
        "학사일정 전체 어디서 봐요?",
        "이번 학기 시험 기간 알려줘",
        "성적 발표 언제 해요?",
        "계절학기 신청 기간이 언제예요?",
        "휴학 신청 언제까지예요?",
        "개강 첫날이 언제야?",
        "이번 학기 학사일정 전체 보여줘",
        "수강 취소 기간 언제까지예요?",
        "성적 이의신청 기간은 언제예요?",
        "복학 신청 기간 알려줘",
    ],

    # 3: 식단안내
    3: [
        "학식 머나와",
        "밥 뭐 나와요?",
        "오늘 메뉴 알려줘",
        "학생회관 밥 뭐임?",
        "내일 학식 뭐 나와?",
        "점심 뭐 먹을 수 있어요?",
        "구내식당 오늘 뭐 팔아?",
        "저녁 식단 알려줘",
        "이번주 학식 메뉴 궁금해",
        "식당 몇 시에 열어요?",
        "제1학생회관 오늘 점심 뭐야?",
        "다음주 식단도 나와 있어요?",
        "오늘 학교 밥 뭐임?",
        "학식 가격 알려줘",
        "조식 메뉴 뭐예요?",
        "지난주 식단 어땠어?",
        "이번주 수요일 식단 알려줘",
    ],

    # 4: 통학/셔틀버스
    4: [
        "셔틀 언제와?",
        "버스 몇 시에 있어요?",
        "셧틀 시간 알려줘",
        "대덕캠퍼스 가는 버스 있어?",
        "통학버스 노선 알려줘",
        "스쿨버스 운행 시간 알려줘",
        "교내순환버스 언제 다녀?",
        "캠퍼스 순환버스 타는 곳 어디야?",
        "보운 가는 셔틀 있어?",
        "첫차 몇 시예요?",
        "막차 시간 알려줘",
        "셔틀 오늘 운행해요?",
        "통학 버스 어디서 타요?",
        "버스 배차 간격이 어떻게 돼요?",
        "셔틀버스 방학에도 운행해요?",
        "캠퍼스 버스 정류장 어디 있어요?",
        "셔틀 타려면 어디 가야 해요?",
    ],
}


def augment(dry: bool = False):
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    before = dict(sorted(Counter(d["label"] for d in train).items()))

    # 기존 질문 set (중복 방지)
    existing = {d["question"] for d in train}

    added = 0
    for label, questions in _AUGMENT.items():
        for q in questions:
            if q not in existing:
                train.append({"question": q, "label": label})
                existing.add(q)
                added += 1

    after = dict(sorted(Counter(d["label"] for d in train).items()))

    print(f"Before: {before}  total={sum(before.values())}")
    print(f"After : {after}  total={sum(after.values())}")
    print(f"Added : {added}개")

    if dry:
        print("[DRY RUN] 파일 저장 안 함")
        return

    # 백업 후 저장
    backup = TRAIN_PATH.parent / "train_aug_backup.json"
    backup.write_bytes(TRAIN_PATH.read_bytes())
    TRAIN_PATH.write_text(
        json.dumps(train, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"저장 완료: {TRAIN_PATH}")
    print(f"백업    : {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="미리보기만 (파일 변경 없음)")
    args = parser.parse_args()
    augment(dry=args.dry)
