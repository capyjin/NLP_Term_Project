# Colab 제출 전 실측 테스트

정규 제출 경로:

```text
/content/drive/MyDrive/NLP_Termproject/Termproject_박연진
```

## 1. Task 1 Run All

1. Colab에서 `classifier_박연진.ipynb`를 연다.
2. 런타임 유형을 T4 GPU로 설정한다.
3. `런타임 > 모두 실행`을 실행한다.
4. 셀 1에서 Drive 연결을 승인한다.

확인 항목:

- 셀 1이 `requirements.txt`에서 `kiwipiepy`를 포함한 Task1/2/3 필수 패키지를 한 번에 설치한다.
- 런타임 재시작 안내 없이 다음 셀로 진행한다.
- 프로젝트 루트가 정규 제출 경로로 출력된다.
- 마지막에 `outputs/cls_output.json`이 생성된다.

## 2. Task 2/3 실행

Colab 터미널에서 실행:

```bash
cd /content/drive/MyDrive/NLP_Termproject/Termproject_박연진
bash chatbot.sh
```

확인 항목:

- 노트북 셀 1을 건너뛰어도 `[0/4] 실행 환경 확인 중...` 단계에서 Task2/3 필수 패키지를 설치한다.
- `outputs/chat_output.json`이 생성된다.
- `data/test_realtime.json`이 있으면 `outputs/realtime_output.json`이 생성된다.
- Gradio 접속 URL이 출력된다.
- 모델 캐시가 없으면 Hugging Face 다운로드로 인해 첫 실행에 약 10분 이상 걸릴 수 있다.

## 3. 결과 파일 확인

```bash
ls -l outputs/cls_output.json outputs/chat_output.json
test -f outputs/realtime_output.json && ls -l outputs/realtime_output.json
```

각 JSON 파일은 질문별 결과 배열이어야 하며 빈 파일이면 안 된다.

## 4. 실패 시 체크포인트

- 경로 오류: Drive에 `Termproject_박연진/requirements.txt`, `chatbot.sh`, `src/`, `data/`가 바로 존재하는지 확인한다.
- Git pull 오류: 제출 폴더에 미커밋 변경이 있는지 확인한다. pull 실패는 경고 후 현재 제출 코드로 계속 진행한다.
- 패키지 오류: 셀 1 출력에서 누락된 Task1 패키지명을 확인한다.
- GPU 메모리 오류: T4 GPU 연결 여부를 확인하고 런타임을 초기화한 뒤 다시 실행한다.
