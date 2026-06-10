# 다음 에이전트 인수인계

작성 시각: 2026-06-10  
프로젝트: 충남대학교 자연어처리 텀프로젝트 CNU Campus ChatBot  
함께 전달할 문서: `PROFESSOR_SUBMISSION_REVIEW.md`

## 1. 목표와 중단 조건

최우선 목표는 새 기능 추가가 아니라 **조교가 실행하는 두 진입점의 clean-run 성공과 최종 제출물 완성**이다.

- Task 1 진입점: `classifier_박연진.ipynb`
- Task 2 진입점: `bash chatbot.sh`
- 제출 기한: `2026-06-12 24:00 KST`
- 완료 판정: 새 Colab T4 환경에서 두 진입점이 수동 코드 수정 없이 끝까지 실행되고, 실제 출력 JSON과 UI 동작 증거가 있으며, 발표자료·영상·최종 ZIP이 준비된 상태

`PROFESSOR_SUBMISSION_REVIEW.md`는 최초 교수 관점 정적 검토 결과다. 그 문서의 일부 지적은 이후 코드로 보완 중이므로, 아래 최신 상태를 우선해서 판단하라.

## 2. Git 및 작업 트리 상태

- 브랜치: `main`
- 원격 대비: `origin/main`보다 1커밋 앞섬
- 최신 커밋: `4bbb056 fix: 분류기 노트북 경로 위치 독립화 (Phase A)`
- 작업 트리는 매우 dirty하다. **기존 수정·삭제·미추적 파일을 임의로 되돌리거나 정리하지 말 것.**
- `.omc`, `.omx`, `__pycache__`, 데이터 JSON 변경은 사용자/이전 에이전트 산출물이 섞여 있다. 제출 ZIP 정리는 별도 제출 디렉터리에서 하는 편이 안전하다.

최근 커밋 이후에도 `classifier_박연진.ipynb`와 `scripts/make_notebook.py`에 미커밋 개선이 있다.

- 고정 시드 `SEED=42`
- `FORCE_RETRAIN=True`
- train/valid 질문 중복 감지 후 누수 없는 stratified split 사용
- DataLoader 셔플 시드 고정

## 3. 현재까지 구현된 보완

### Phase A: 분류기 노트북

- 특정 Google Drive 경로 하드코딩을 제거하고 프로젝트 루트 자동 탐색을 추가했다.
- 현재 미커밋 버전은 검증셋 누수 감지와 재분리까지 포함한다.
- 아직 새 Colab T4 런타임에서 전체 셀 실행, macro F1, 실제 `cls_output.json` 생성은 검증하지 못했다.

### Phase B: 벡터 DB 정합성

다음 코드가 작성되었으나 대부분 미커밋 상태다.

- `src/embedding/config.py`: KURE-v1 / 1024차원 계약
- `src/vectordb/integrity.py`: manifest와 DB ID 정합성 검사
- `scripts/check_vector_db.py`: 정합성 CLI
- `src/vectordb/build_db.py --fresh`: 고정 프로젝트 DB를 삭제하고 전체 재구축
- `chatbot.sh`, `chatbot.bat`: FAQ 반영 후 정합성 검사, 불일치 시 자동 재구축
- `tests/test_vector_db_integrity.py`: 정합성 회귀 테스트

중요: **검사와 자동 재구축 로직은 구현됐지만 실제 패키지 DB는 아직 재구축되지 않았다.**

현재 읽기 전용 검사 결과:

```text
[INVALID] 벡터 DB 재구축이 필요합니다.
  - build_manifest.json이 없습니다.
  - 색인 ID 불일치: DB 158건 / chunks.json 205건
```

### 공지 미래 날짜 보완

- `src/handlers/notice_handler.py`에서 `등록일`, `접수기간`, 일반 ISO 날짜, 점 형식 날짜의 유효성 및 미래 날짜 제외 로직을 보완했다.
- `tests/test_notice_date.py`가 추가됐다.
- 기존 `outputs/realtime_output.json`은 수정 전 산출물이므로 여전히 `2026-12-31`을 최신 게시일로 표시한다. 코드 수정 완료와 출력 품질 해결을 동일시하면 안 된다.

### 문서 정합성

- README의 기본 생성 모델 설명을 실제 기본값인 `Qwen2.5-3B-Instruct`와 맞췄다.
- 모델 다운로드 링크와 벡터 DB 자동 검사 흐름을 추가했다.

## 4. 2026-06-10 로컬 검증 결과

통과:

```powershell
py -3.12 -m unittest tests.test_vector_db_integrity tests.test_notice_date -v
```

- 7 tests, 모두 통과

```powershell
$env:PYTHONIOENCODING='utf-8'; py -3.12 tests\verify_pipeline.py
```

- PASS 31 / FAIL 0 / SKIP 5
- `chunks.json`: 205건, 중복 ID 0
- SKIP: `kiwipiepy` 미설치 BM25 실제 검색, ChromaDB 구축, 1024d 실제 검증, Qwen 생성, end-to-end 답변

미통과 또는 미검증:

- `py -3.12 scripts\check_vector_db.py`: 실패, DB 158건 / chunks 205건
- 새 Colab T4에서 분류기 전체 실행: 미검증
- `bash chatbot.sh` 전체 clean-run: 미검증
- 실제 Qwen 응답 생성 및 UI 동작: 미검증
- 실시간 출력 재생성 후 미래 날짜 제거 확인: 미검증

현재 출력 상태:

- `outputs/chat_output.json`: 2026-06-01 생성, 5건 모두 `[MOCK]`
- `outputs/cls_output.json`: 존재하지만 최신 clean-run 증거 아님
- `outputs/realtime_output.json`: 2026-06-09 생성, 미래 날짜 및 셔틀 서식 문제 포함
- `outputs/eval_result.json`: 라우팅 100/100 정답이지만 모델 응답 품질 검증은 아님

## 5. 다음 에이전트가 바로 할 일

다음 순서를 유지하라. DB가 불일치한 상태에서 `chatbot.sh` 성공을 판정하지 말 것.

1. 현재 미커밋 코드 리뷰
   - 특히 `classifier_박연진.ipynb`, `scripts/make_notebook.py`, `src/vectordb/integrity.py`, `src/vectordb/build_db.py`, `chatbot.sh`
   - 기존 사용자 변경을 보존하고 관련 변경만 수정할 것

2. 로컬에서 가능한 검증 확대
   - 전체 관련 단위 테스트 실행
   - 가능하면 `kiwipiepy`, `chromadb` 환경에서 검색/DB 검사
   - 자동 재구축 실패 시 원인을 수정하되 새 의존성은 추가하지 말 것

3. GPU/Colab에서 Phase A 증명
   - 새 Colab T4 런타임에서 `classifier_박연진.ipynb` 셀 1~8 순차 실행
   - 수동 경로 수정 없이 실행되는지 확인
   - macro F1, 입력/출력 개수, `outputs/cls_output.json`을 증거로 남길 것

4. GPU/Colab에서 Phase B/C 증명
   - `python scripts/check_vector_db.py`가 불일치를 보고하는지 확인
   - `python src/vectordb/build_db.py --fresh`로 재구축
   - `chunks=205`, DB IDs `=205`, manifest 존재, 실제 임베딩 1024d를 확인
   - 완전히 새 런타임에서 `bash chatbot.sh` 한 번으로 실제 JSON 생성과 UI 실행까지 확인
   - `outputs/chat_output.json`에 `[MOCK]`가 없어야 한다

5. 실시간 출력 재검증
   - `python src/realtime_model.py --input data/test_realtime.json --output outputs/realtime_output.json`
   - “가장 최근에 올라온 공지사항은 언제 게시되었나요?”에 오늘 이후 날짜가 게시일로 나오지 않는지 확인
   - 셔틀 정류장 출력의 빈 괄호·줄바꿈 훼손도 확인

6. 제출 패키징
   - 발표자료, 약 2분 UI 영상, 모델 파일 또는 접근 가능한 다운로드 링크 준비
   - 최종 파일명: `Termproject_박연진.zip`
   - `.git`, `.omc`, `.omx`, `__pycache__`, 중복/실험 노트북, 오래된 MOCK 출력은 최종 ZIP에서 제외
   - 별도 위치에 ZIP을 풀어 README 절차대로 다시 확인

## 6. 우선순위와 범위 제한

P0:

- 분류기 clean-run
- DB 205/205 및 1024d 정합성
- `chatbot.sh` clean-run
- MOCK 없는 실제 출력
- UI 동작
- 발표자료·영상·최종 ZIP

P1:

- 실시간 공지 날짜·셔틀 서식 품질
- 검증셋 누수 없는 실제 F1 기록
- README와 실제 실행 시간/환경 일치

현재 마감 전에는 하지 않아도 되는 것:

- Task 1 학습 분류기를 Task 2 라우터에 직접 연결
- API와 UI/CLI 응답 경로 통합
- 대규모 기능 추가 또는 UI 재설계

## 7. 주의사항

- `PROFESSOR_SUBMISSION_REVIEW.md`의 “분류기 경로 하드코딩” 지적은 최신 커밋과 미커밋 코드에서 보완 중이다. 다만 Colab clean-run 전까지 완료로 판정하지 말 것.
- `chatbot.sh`는 마지막에 UI 서버를 실행하므로 정상적으로 계속 실행 중인 상태가 성공일 수 있다. JSON 생성과 UI 질문/응답을 확인한 뒤 종료할 것.
- `outputs/eval_result.json`의 라우팅 100%는 키워드 라우터만 검증한다. RAG 검색과 생성 답변 품질 증거로 사용하지 말 것.
- 현재 작업 트리에는 자동 생성 파일과 사용자 변경이 섞여 있다. `git reset --hard`, `git checkout --`, 광범위 삭제를 하지 말 것.
- 커밋이 필요하면 프로젝트 `AGENTS.md`의 Lore Commit Protocol을 따를 것.

## 8. 핵심 명령

```powershell
git status --short --branch
py -3.12 -m unittest tests.test_vector_db_integrity tests.test_notice_date -v
$env:PYTHONIOENCODING='utf-8'; py -3.12 tests\verify_pipeline.py
py -3.12 scripts\check_vector_db.py
```

```bash
python src/vectordb/build_db.py --fresh
python scripts/check_vector_db.py
bash chatbot.sh
python src/realtime_model.py --input data/test_realtime.json --output outputs/realtime_output.json
```

## 9. 최종 인계 요약

현재 프로젝트는 기능 추가 단계가 아니라 **제출 재현성 고정 단계**다. 분류기 경로·검증 누수, DB 정합성 자동검사, 미래 날짜 필터 등 핵심 보완 코드는 작성됐고 로컬 단위/정적 테스트는 통과했다. 그러나 실제 DB는 아직 `158/205`로 불일치하며, `chat_output.json`은 전부 MOCK이고, GPU clean-run과 최종 제출물은 아직 증명되지 않았다. 다음 에이전트는 기존 미커밋 작업을 보존하면서 DB 재구축과 두 평가 진입점의 새 환경 종단 검증을 최우선으로 마무리해야 한다.
