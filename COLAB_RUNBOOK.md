# Colab 실행 가이드 — Phase B/C clean-run

> 새 T4 런타임에서 아래 셀을 **순서대로** 실행하세요.  
> 각 셀은 독립된 Colab 코드 셀입니다. 복붙 후 실행하면 됩니다.

---

## 셀 0 — 드라이브 마운트 + git pull

```python
# ── 셀 0: 드라이브 마운트 + 최신 코드 pull ──
from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"  # 실제 폴더명 다르면 수정

result = subprocess.run(
    ["git", "-C", PROJECT, "pull"],
    capture_output=True, text=True
)
print(result.stdout or "(이미 최신)")
print(result.stderr or "")

# 최신 커밋 확인
result2 = subprocess.run(
    ["git", "-C", PROJECT, "log", "--oneline", "-3"],
    capture_output=True, text=True
)
print(result2.stdout)
```

---

## 셀 1 — 패키지 설치 (처음 1회, 설치 후 런타임 재시작)

```python
# ── 셀 1: 패키지 설치 ──
# 실행 후 "런타임 > 런타임 다시 시작" 한 번만 하면 됩니다.
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.47.1", "tokenizers==0.21.0",
    "sentence-transformers==3.0.1", "chromadb==0.5.5",
    "kiwipiepy", "gradio==4.42.0",
    "torch==2.5.1", "bitsandbytes",
    "datasets", "scikit-learn", "tqdm",
], check=True)
print("설치 완료 — 런타임 > 런타임 다시 시작 후 셀 2부터 실행하세요")
```

---

## 셀 2 — 벡터 DB 정합성 검사 + 필요 시 재구축 (Phase B)

```python
# ── 셀 2: 벡터 DB 정합성 확인 및 재구축 ──
import subprocess, sys, os
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# 정합성 검사
r = subprocess.run([sys.executable, "scripts/check_vector_db.py"],
                   capture_output=True, text=True)
print(r.stdout); print(r.stderr)

if r.returncode != 0:
    print("\n[재구축 시작] chunks=205, dim=1024 기준으로 DB를 새로 만듭니다 (~10분)...")
    r2 = subprocess.run([sys.executable, "src/vectordb/build_db.py", "--fresh"],
                        capture_output=True, text=True)
    print(r2.stdout[-3000:])
    if r2.returncode != 0:
        print("ERROR:", r2.stderr[-1000:])
    else:
        # 재검사
        r3 = subprocess.run([sys.executable, "scripts/check_vector_db.py"],
                            capture_output=True, text=True)
        print("\n[재검사]", r3.stdout); print(r3.stderr)
        if r3.returncode == 0:
            print("✓ DB 정합성 확인 완료")
        else:
            print("❌ DB 재구축 후에도 불일치 — 오류 확인 필요")
```

**완료 조건:** `[VALID] chunks=205, embeddings=205, dim=1024` 출력

---

## 셀 3 — classifier_박연진.ipynb 실행 (Phase A 증거 확보)

> 이 셀 대신 **직접 `classifier_박연진.ipynb`를 열고 셀 1~8을 순서대로 실행**하세요.  
> 경로는 자동 탐색되므로 수동 수정 불필요합니다.  
> 완료 후 아래 검증 셀로 결과를 확인하세요.

```python
# ── 셀 3: classifier 결과 검증 ──
import json, os
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"

out = f"{PROJECT}/outputs/cls_output.json"
if not os.path.exists(out):
    print("❌ cls_output.json 없음 — classifier_박연진.ipynb 먼저 실행하세요")
else:
    data = json.load(open(out, encoding="utf-8"))
    print(f"✓ cls_output.json: {len(data)}건")
    labels = [d['label'] for d in data]
    print("라벨 분포:", dict(sorted(
        __import__('collections').Counter(labels).items()
    )))
    print("샘플 3건:")
    for d in data[:3]:
        print(f"  Q: {d['question'][:40]} → label {d['label']}")
```

---

## 셀 4 — chatbot.sh clean-run (Phase C)

```python
# ── 셀 4: chatbot.sh 종단 실행 ──
# UI는 백그라운드로 띄우고 JSON 출력을 먼저 확인합니다.
import subprocess, sys, os, time, json
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"
os.chdir(PROJECT)

# chatbot.sh에서 UI 앞 단계(DB검사+크롤링+chat_output 생성)만 먼저 실행
steps = [
    [sys.executable, "scripts/inject_faq.py"],
    [sys.executable, "scripts/check_vector_db.py"],
    [sys.executable, "src/crawling/meal_crawler.py"],
    [sys.executable, "src/crawling/shuttle_crawler.py"],
    [sys.executable, "src/chatbot_model.py",
     "--input", "data/test_chat.json",
     "--output", "outputs/chat_output.json"],
]
labels = ["inject_faq", "check_db", "meal_crawl", "shuttle_crawl", "chatbot_model"]

for label, cmd in zip(labels, steps):
    r = subprocess.run(cmd, capture_output=True, text=True)
    status = "✓" if r.returncode == 0 else "⚠"
    print(f"{status} {label}")
    if r.returncode != 0 and label not in ("meal_crawl", "shuttle_crawl"):
        print(r.stderr[-500:])

# realtime 있으면 처리
if os.path.exists("data/test_realtime.json"):
    r = subprocess.run([sys.executable, "src/realtime_model.py",
                        "--input", "data/test_realtime.json",
                        "--output", "outputs/realtime_output.json"],
                       capture_output=True, text=True)
    print("✓ realtime_model" if r.returncode == 0 else f"⚠ realtime: {r.stderr[-300:]}")
```

---

## 셀 5 — 출력 검증 (MOCK 없는지, 미래날짜 없는지)

```python
# ── 셀 5: 출력 파일 최종 검증 ──
import json, os
from datetime import date
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"

today = date.today().isoformat()

# chat_output 검증
chat_path = f"{PROJECT}/outputs/chat_output.json"
chat = json.load(open(chat_path, encoding="utf-8"))
mock_count = sum(1 for d in chat if "[MOCK]" in str(d.get("model", "")))
print(f"chat_output.json: {len(chat)}건, MOCK={mock_count} {'✓' if mock_count==0 else '❌'}")
for d in chat:
    print(f"  Q: {d.get('user','')[:40]}")
    print(f"  A: {str(d.get('model',''))[:80]}")
    print()

# realtime_output 검증 (있으면)
rt_path = f"{PROJECT}/outputs/realtime_output.json"
if os.path.exists(rt_path):
    rt = json.load(open(rt_path, encoding="utf-8"))
    print(f"\nrealtime_output.json: {len(rt)}건")
    for d in rt:
        ans = str(d.get("model", ""))
        # 미래 날짜 감지
        import re
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", ans)
        future = [dt for dt in dates if dt > today]
        flag = f"⚠ 미래날짜:{future}" if future else "✓"
        print(f"  {flag} Q: {d.get('user','')[:40]}")
```

---

## 셀 6 — Gradio UI 실행

```python
# ── 셀 6: Gradio UI 실행 ──
import subprocess, sys, os
PROJECT = "/content/drive/MyDrive/NLP_Term_Project"
os.chdir(PROJECT)
subprocess.Popen([sys.executable, "src/chatbot_ui.py"])
print("UI 실행 중... share=True 공개 URL이 곧 출력됩니다.")
print("UI에서 질문 입력 → 응답 확인 후 이 셀을 중단하세요.")
```

---

## 완료 체크리스트

- [ ] 셀 2: `chunks=205, embeddings=205, dim=1024` 확인
- [ ] 셀 3: `cls_output.json` 생성, macro F1 출력
- [ ] 셀 5: `chat_output.json` MOCK=0
- [ ] 셀 5: `realtime_output.json` 미래날짜 없음
- [ ] 셀 6: UI에서 질문 입력 → 응답 정상 출력
