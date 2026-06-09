#!/usr/bin/env python3
"""
Kaggle 메인 노트북 자동 생성 스크립트.

실행:
    python scripts/make_kaggle_notebook.py
출력:
    notebooks/kaggle_main.ipynb

이 스크립트가 생성한 노트북을 Kaggle에 업로드하거나
GitHub에서 clone 후 열어서 셀 순서대로 실행하세요.
"""

import json
from pathlib import Path


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


# ── 셀 정의 ──────────────────────────────────────────────────────────────────

cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# NLP Term Project — Kaggle 실행 노트북\n\n"
    "**과제 지정 환경: Python 3.10.12 / torch 2.5.1**\n\n"
    "## 실행 순서\n\n"
    "| 셀 | 내용 | 비고 |\n"
    "|---|---|---|\n"
    "| Cell 1 | 환경 확인 | Python/GPU 상태 점검 |\n"
    "| Cell 2 | torch 2.5.1 설치 | **실행 후 커널 재시작 필수** |\n"
    "| Cell 3 | 패키지 설치 | 재시작 후 실행 |\n"
    "| Cell 4 | 프로젝트 경로 설정 | GitHub clone 포함 |\n"
    "| Cell 5 | 데이터 파일 확인 | |\n"
    "| Cell 6 | refresh_data | 식단·셔틀·학사일정 갱신 |\n"
    "| Cell 7 | ChromaDB 구축 | 2~5분 소요 |\n"
    "| Cell 8 | Task 2 | chat_output.json 생성 |\n"
    "| Cell 9 | Optional | realtime_output.json 생성 |\n"
    "| Cell 10 | Task 1 | cls_output.json 생성 |\n"
    "| Cell 11 | Gradio UI | 공개 URL 출력 |\n"
    "| Cell 12 | Output 검증 | 최종 확인 |\n"
    "| Cell 13 | requirements.txt | pip freeze → git push |\n\n"
    "> ⚠️ **중요**: Cell 2 실행 후 반드시 `Run > Restart Session` 후 Cell 3부터 실행하세요.",
    "00-header",
))

# ── Cell 1: 환경 확인 ─────────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 1: 환경 확인 ─────────────────────────────────────────────\n"
    "# Python 버전, GPU, CUDA 상태를 확인합니다.\n"
    "# 과제 지정: Python 3.10.12 / torch 2.5.1\n"
    "import sys, subprocess\n"
    "\n"
    "print('=' * 60)\n"
    "print(f'Python : {sys.version}')\n"
    "print(f'Platform: {sys.platform}')\n"
    "\n"
    "# GPU 확인 (nvidia-smi)\n"
    "r = subprocess.run(\n"
    "    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',\n"
    "     '--format=csv,noheader'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "if r.returncode == 0:\n"
    "    print(f'GPU    : {r.stdout.strip()}')\n"
    "else:\n"
    "    print('GPU    : 없음 — Kaggle Accelerator(T4/P100) 설정 확인 필요')\n"
    "\n"
    "# torch 현재 버전\n"
    "try:\n"
    "    import torch\n"
    "    print(f'torch  : {torch.__version__}')\n"
    "    print(f'CUDA   : {torch.version.cuda}')\n"
    "    print(f'GPU OK : {torch.cuda.is_available()}')\n"
    "except ImportError:\n"
    "    print('torch  : 미설치 — Cell 2 실행 필요')\n"
    "\n"
    "# Python 버전 경고\n"
    "ver = sys.version_info\n"
    "if (ver.major, ver.minor) == (3, 10):\n"
    "    print('OK   Python 3.10.x — 과제 지정 범위 내')\n"
    "else:\n"
    "    print(f'WARN Python {sys.version.split()[0]} — 과제 지정: 3.10.12')\n"
    "    print('     코드 동작 가능. requirements.txt에 실제 버전 기록됨.')\n"
    "print('=' * 60)",
    "cell-01",
))

# ── Cell 2: torch 2.5.1 설치 ──────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 2: torch 2.5.1 설치 ──────────────────────────────────────\n"
    "# 실행 후 반드시 커널을 재시작하고 Cell 3부터 이어서 실행하세요!\n"
    "import sys, subprocess\n"
    "\n"
    "try:\n"
    "    import torch\n"
    "    current = torch.__version__\n"
    "except ImportError:\n"
    "    current = 'not_installed'\n"
    "\n"
    "print(f'현재 torch: {current}')\n"
    "\n"
    "if current.startswith('2.5.1'):\n"
    "    print('OK  torch 2.5.1 이미 설치됨 — 커널 재시작 불필요, Cell 3 진행')\n"
    "else:\n"
    "    # CUDA 버전 자동 감지 (Kaggle 기본: CUDA 12.1)\n"
    "    r = subprocess.run(\n"
    "        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],\n"
    "        capture_output=True, text=True\n"
    "    )\n"
    "    # Kaggle 2025~2026 기본 CUDA 12.1 → cu121\n"
    "    cuda_tag = 'cu121'\n"
    "    print(f'CUDA wheel: {cuda_tag}')\n"
    "    print(f'torch==2.5.1+{cuda_tag} 설치 중 (1~3분)...')\n"
    "\n"
    "    result = subprocess.run([\n"
    "        sys.executable, '-m', 'pip', 'install',\n"
    "        f'torch==2.5.1+{cuda_tag}',\n"
    "        f'torchvision==0.20.1+{cuda_tag}',\n"
    "        f'torchaudio==2.5.1+{cuda_tag}',\n"
    "        '--index-url', f'https://download.pytorch.org/whl/{cuda_tag}',\n"
    "        '--quiet',\n"
    "    ])\n"
    "\n"
    "    if result.returncode == 0:\n"
    "        print()\n"
    "        print('설치 완료!')\n"
    "        print('★' * 52)\n"
    "        print('  커널 재시작이 필요합니다!')\n"
    "        print('  Kaggle: Run > Restart Session 클릭')\n"
    "        print('  재시작 후 Cell 3부터 순서대로 실행하세요.')\n"
    "        print('★' * 52)\n"
    "    else:\n"
    "        print('FAIL 설치 실패')\n"
    "        print('cu124로 재시도: torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124')",
    "cell-02",
))

# ── Restart notice ────────────────────────────────────────────────────────────
cells.append(md_cell(
    "## ⚠️ 커널 재시작 후 Cell 3부터 실행\n\n"
    "Cell 2에서 torch 2.5.1 설치를 완료했다면:\n\n"
    "1. Kaggle 상단 메뉴 **Run > Restart Session** 클릭\n"
    "2. 팝업에서 **Restart** 확인\n"
    "3. **Cell 3부터** 순서대로 실행\n\n"
    "> torch가 이미 2.5.1이면 재시작 없이 바로 Cell 3으로 진행하세요.",
    "restart-notice",
))

# ── Cell 3: 패키지 설치 ───────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 3: 패키지 설치 ────────────────────────────────────────────\n"
    "# 커널 재시작 후 이 셀부터 실행하세요.\n"
    "# torch는 이미 설치됐으므로 덮어쓰지 않습니다.\n"
    "import sys, subprocess\n"
    "\n"
    "packages = [\n"
    "    # Transformers / 양자화\n"
    "    'transformers==4.44.2', 'accelerate==0.34.2', 'bitsandbytes==0.43.3',\n"
    "    'peft==0.12.0', 'datasets==2.21.0', 'trl==0.9.6',\n"
    "    # RAG / VectorDB\n"
    "    'sentence-transformers==3.0.1', 'chromadb==0.5.5',\n"
    "    'langchain==0.2.16', 'langchain-community==0.2.17',\n"
    "    'rank-bm25==0.2.2', 'kiwipiepy==0.17.1',\n"
    "    # 크롤링\n"
    "    'beautifulsoup4==4.12.3', 'requests==2.32.3',\n"
    "    'lxml==5.3.0', 'pdfplumber==0.11.4',\n"
    "    # 분류기 / UI / 유틸\n"
    "    'scikit-learn==1.5.1', 'gradio==4.42.0',\n"
    "    'tqdm==4.66.5', 'python-dotenv==1.0.1',\n"
    "]\n"
    "\n"
    "for pkg in packages:\n"
    "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'],\n"
    "                       capture_output=True)\n"
    "    print(f'{'OK  ' if r.returncode == 0 else 'FAIL'} {pkg}')\n"
    "\n"
    "# torch 2.5.1 보호 확인 (다른 패키지가 덮어쓰는지 체크)\n"
    "import torch\n"
    "tv = torch.__version__\n"
    "if tv.startswith('2.5.1'):\n"
    "    print(f'OK   torch {tv} 유지됨 (과제 지정 버전)')\n"
    "else:\n"
    "    print(f'WARN torch {tv} — 버전이 변경됨. Cell 2 재실행 후 커널 재시작 필요.')",
    "cell-03",
))

# ── Cell 4: 프로젝트 경로 설정 ────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 4: 프로젝트 경로 설정 ─────────────────────────────────────\n"
    "# 프로젝트 루트를 자동으로 감지하고, 없으면 GitHub에서 clone합니다.\n"
    "import sys, os, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "# GitHub remote URL 자동 감지 (git config에서)\n"
    "r = subprocess.run(['git', 'remote', 'get-url', 'origin'],\n"
    "                   capture_output=True, text=True)\n"
    "REPO_URL = r.stdout.strip() if r.returncode == 0 else \\\n"
    "           'https://github.com/여기에본인아이디/NLP_Term_Project.git'\n"
    "print(f'REPO: {REPO_URL}')\n"
    "\n"
    "# 프로젝트 루트 자동 감지\n"
    "candidates = [\n"
    "    Path('/kaggle/working/NLP_Term_Project'),\n"
    "    Path('/kaggle/working'),\n"
    "    Path('/content/NLP_Term_Project'),\n"
    "    Path('/content/drive/MyDrive/NLP_Term_Project'),\n"
    "    Path.cwd(),\n"
    "]\n"
    "ROOT = None\n"
    "for p in candidates:\n"
    "    if (p / 'src').exists() and (p / 'data').exists():\n"
    "        ROOT = p.resolve()\n"
    "        break\n"
    "\n"
    "# 없으면 GitHub clone\n"
    "if ROOT is None:\n"
    "    target = Path('/kaggle/working/NLP_Term_Project')\n"
    "    print(f'프로젝트 없음 — clone 중: {REPO_URL}')\n"
    "    os.system(f'git clone {REPO_URL} {target}')\n"
    "    ROOT = target.resolve()\n"
    "else:\n"
    "    # 최신 코드 pull\n"
    "    print('git pull 중...')\n"
    "    os.system(f'git -C {ROOT} pull origin main')\n"
    "\n"
    "os.chdir(ROOT)\n"
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n"
    "(ROOT / 'outputs').mkdir(exist_ok=True)\n"
    "\n"
    "print(f'PROJECT ROOT : {ROOT}')\n"
    "print(f'outputs/     : {ROOT / \"outputs\"}')\n"
    "print(f'cwd          : {os.getcwd()}')",
    "cell-04",
))

# ── Cell 5: 데이터 파일 확인 ──────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 5: 데이터 파일 확인 ───────────────────────────────────────\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "required = {\n"
    "    'data/test_cls.json':           'Task1 테스트',\n"
    "    'data/test_chat.json':          'Task2 테스트',\n"
    "    'data/test_realtime.json':      'Optional 테스트',\n"
    "    'data/train.json':              '학습 데이터',\n"
    "    'data/valid.json':              '검증 데이터',\n"
    "    'data/processed/chunks.json':   'RAG 청크',\n"
    "    'src/chatbot_model.py':         'Task2 스크립트',\n"
    "    'src/realtime_model.py':        'Optional 스크립트',\n"
    "    'src/chatbot_ui.py':            'Gradio UI',\n"
    "    'classifier_박연진.ipynb':       'Task1 분류기',\n"
    "}\n"
    "\n"
    "missing = []\n"
    "for path, desc in required.items():\n"
    "    p = Path(path)\n"
    "    if p.exists():\n"
    "        if path.endswith('.json'):\n"
    "            with open(p, encoding='utf-8') as f:\n"
    "                d = json.load(f)\n"
    "            n = len(d) if isinstance(d, list) else 'dict'\n"
    "            print(f'OK   {path} ({n}건)')\n"
    "        else:\n"
    "            print(f'OK   {path}')\n"
    "    else:\n"
    "        print(f'MISS {path}  ← {desc}')\n"
    "        missing.append(path)\n"
    "\n"
    "print()\n"
    "if missing:\n"
    "    print(f'누락 파일 {len(missing)}개 — git pull 또는 데이터 확인 필요')\n"
    "else:\n"
    "    print('OK  모든 필수 파일 확인됨')",
    "cell-05",
))

# ── Cell 6: refresh_data ──────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 6: 실시간 데이터 갱신 ─────────────────────────────────────\n"
    "# 식단, 셔틀버스, 학사일정, 공지사항 최신 데이터를 크롤링합니다.\n"
    "import os\n"
    "\n"
    "print('데이터 갱신 중...')\n"
    "ret = os.system('python scripts/refresh_data.py')\n"
    "\n"
    "if ret == 0:\n"
    "    print('OK  refresh_data 완료')\n"
    "else:\n"
    "    print(f'WARN refresh_data 일부 실패 (exit {ret}) — 기존 데이터로 진행 가능')",
    "cell-06",
))

# ── Cell 7: ChromaDB 구축 ─────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 7: ChromaDB 구축 (2~5분 소요) ────────────────────────────\n"
    "# 이미 chroma_db가 있으면 건너뜁니다.\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "chroma_path = Path('chroma_db')\n"
    "files = list(chroma_path.iterdir()) if chroma_path.exists() else []\n"
    "\n"
    "if files:\n"
    "    print(f'chroma_db 존재 ({len(files)}개 파일) — 구축 건너뜀')\n"
    "    print('재구축: import shutil; shutil.rmtree(\"chroma_db\") 후 이 셀 재실행')\n"
    "else:\n"
    "    print('ChromaDB 구축 중 (2~5분)...')\n"
    "    ret1 = os.system('python scripts/inject_faq.py --reset')\n"
    "    ret2 = os.system('python src/vectordb/build_db.py')\n"
    "    if ret1 == 0 and ret2 == 0:\n"
    "        files_after = list(chroma_path.iterdir()) if chroma_path.exists() else []\n"
    "        print(f'OK  ChromaDB 구축 완료 ({len(files_after)}개 파일)')\n"
    "    else:\n"
    "        print('FAIL ChromaDB 구축 실패 — 오류 메시지 확인 필요')",
    "cell-07",
))

# ── Cell 8: Task 2 ─────────────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 8: Task 2 — chat_output.json 생성 ─────────────────────────\n"
    "import os, json\n"
    "from pathlib import Path\n"
    "\n"
    "print('Task 2 실행 중 (Qwen2.5-3B 로드 ~ 3분)...')\n"
    "ret = os.system(\n"
    "    'python src/chatbot_model.py '\n"
    "    '--input data/test_chat.json '\n"
    "    '--output outputs/chat_output.json'\n"
    ")\n"
    "\n"
    "out = Path('outputs/chat_output.json')\n"
    "if ret == 0 and out.exists():\n"
    "    with open(out, encoding='utf-8') as f:\n"
    "        data = json.load(f)\n"
    "    mock = sum(1 for d in data if '[MOCK]' in str(d.get('model', '')))\n"
    "    status = f'WARN MOCK {mock}건 포함' if mock else 'OK  '\n"
    "    print(f'{status} chat_output.json: {len(data)}건')\n"
    "    if mock:\n"
    "        print('     MOCK → ChromaDB 구축 확인 후 재실행 필요')\n"
    "    print()\n"
    "    for item in data[:2]:\n"
    "        print(f'  Q: {item.get(\"user\", item.get(\"question\", \"\"))[:60]}')\n"
    "        print(f'  A: {item.get(\"model\", \"\")[:80]}')\n"
    "        print()\n"
    "else:\n"
    "    print(f'FAIL chat_output.json 생성 실패 (exit {ret})')",
    "cell-08",
))

# ── Cell 9: Optional ──────────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 9: Optional — realtime_output.json 생성 ───────────────────\n"
    "import os, json\n"
    "from pathlib import Path\n"
    "\n"
    "print('Optional 실행 중...')\n"
    "ret = os.system(\n"
    "    'python src/realtime_model.py '\n"
    "    '--input data/test_realtime.json '\n"
    "    '--output outputs/realtime_output.json'\n"
    ")\n"
    "\n"
    "out = Path('outputs/realtime_output.json')\n"
    "if ret == 0 and out.exists():\n"
    "    with open(out, encoding='utf-8') as f:\n"
    "        data = json.load(f)\n"
    "    print(f'OK  realtime_output.json: {len(data)}건')\n"
    "    for item in data:\n"
    "        print(f'  Q: {item.get(\"user\", \"\")[:60]}')\n"
    "        print(f'  A: {item.get(\"model\", \"\")[:80]}')\n"
    "        print()\n"
    "else:\n"
    "    print(f'FAIL realtime_output.json 생성 실패 (exit {ret})')",
    "cell-09",
))

# ── Cell 10: Task 1 ────────────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 10: Task 1 — cls_output.json 생성 ─────────────────────────\n"
    "# classifier_박연진.ipynb 를 nbconvert로 실행합니다.\n"
    "import subprocess, sys, os, json\n"
    "from pathlib import Path\n"
    "\n"
    "# nbconvert 설치 확인\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install',\n"
    "                'jupyter', 'nbconvert', '--quiet'], capture_output=True)\n"
    "\n"
    "nb_path = Path('classifier_박연진.ipynb')\n"
    "if not nb_path.exists():\n"
    "    print('FAIL classifier_박연진.ipynb 없음 — Cell 4(git pull) 확인')\n"
    "else:\n"
    "    print('classifier_박연진.ipynb 실행 중 (KLUE-BERT 학습, 10~30분)...')\n"
    "    result = subprocess.run([\n"
    "        sys.executable, '-m', 'jupyter', 'nbconvert',\n"
    "        '--to', 'notebook',\n"
    "        '--execute',\n"
    "        '--ExecutePreprocessor.timeout=3600',\n"
    "        '--inplace',\n"
    "        str(nb_path),\n"
    "    ], capture_output=True, text=True)\n"
    "\n"
    "    out = Path('outputs/cls_output.json')\n"
    "    if result.returncode == 0 and out.exists():\n"
    "        with open(out, encoding='utf-8') as f:\n"
    "            data = json.load(f)\n"
    "        print(f'OK  cls_output.json: {len(data)}건')\n"
    "        print(f'    sample: {data[0]}')\n"
    "    elif result.returncode == 0:\n"
    "        print('WARN nbconvert 성공했지만 cls_output.json 없음')\n"
    "        print('     classifier_박연진.ipynb 내 outputs 경로 확인 필요')\n"
    "    else:\n"
    "        print('FAIL nbconvert 실패')\n"
    "        print(result.stderr[-800:])",
    "cell-10",
))

# ── Cell 11: Gradio UI ────────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 11: Gradio UI 실행 ────────────────────────────────────────\n"
    "# share=True로 공개 URL을 생성합니다 (https://xxxx.gradio.live).\n"
    "# Ctrl+C 또는 Kaggle Stop 버튼으로 종료.\n"
    "import subprocess, sys, time\n"
    "from pathlib import Path\n"
    "\n"
    "ui_path = Path('src/chatbot_ui.py')\n"
    "code = ui_path.read_text(encoding='utf-8')\n"
    "\n"
    "# share=True 자동 적용\n"
    "if 'share=True' not in code:\n"
    "    code = code.replace('demo.launch(', 'demo.launch(share=True, ')\n"
    "    ui_path.write_text(code, encoding='utf-8')\n"
    "    print('share=True 적용 완료')\n"
    "\n"
    "print('Gradio UI 시작 중... (모델 로드 1~3분 소요)')\n"
    "print('공개 URL(gradio.live)이 출력되면 접속하세요.')\n"
    "print()\n"
    "\n"
    "proc = subprocess.Popen(\n"
    "    [sys.executable, 'src/chatbot_ui.py'],\n"
    "    stdout=subprocess.PIPE,\n"
    "    stderr=subprocess.STDOUT,\n"
    "    text=True, bufsize=1,\n"
    ")\n"
    "\n"
    "for _ in range(240):  # 최대 4분 대기\n"
    "    line = proc.stdout.readline()\n"
    "    if line:\n"
    "        print(line.rstrip())\n"
    "    if 'gradio.live' in line or 'Running on public URL' in line:\n"
    "        print()\n"
    "        print('UI 접속 가능! 위 URL을 브라우저에서 여세요.')\n"
    "        break\n"
    "    time.sleep(1)",
    "cell-11",
))

# ── Cell 12: Output 검증 ──────────────────────────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 12: Output 파일 검증 ──────────────────────────────────────\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "outputs = [\n"
    "    ('outputs/chat_output.json',     'Task 2',   True),\n"
    "    ('outputs/realtime_output.json', 'Optional', True),\n"
    "    ('outputs/cls_output.json',      'Task 1',   False),\n"
    "]\n"
    "\n"
    "print('=' * 60)\n"
    "all_ok = True\n"
    "for path, label, check_mock in outputs:\n"
    "    p = Path(path)\n"
    "    if not p.exists():\n"
    "        print(f'MISS [{label:8}] {path}')\n"
    "        all_ok = False\n"
    "        continue\n"
    "    with open(p, encoding='utf-8') as f:\n"
    "        data = json.load(f)\n"
    "    n = len(data)\n"
    "    if check_mock:\n"
    "        mock = sum(1 for d in data if '[MOCK]' in str(d.get('model', '')))\n"
    "        if mock:\n"
    "            print(f'WARN [{label:8}] {path}: {n}건 (MOCK {mock}건 포함)')\n"
    "        else:\n"
    "            print(f'OK   [{label:8}] {path}: {n}건')\n"
    "    else:\n"
    "        print(f'OK   [{label:8}] {path}: {n}건')\n"
    "print('=' * 60)\n"
    "print('모든 output 정상' if all_ok else '일부 누락 — 해당 셀 재실행 필요')",
    "cell-12",
))

# ── Cell 13: requirements.txt 생성 + git push ────────────────────────────────
cells.append(code_cell(
    "# ─── Cell 13: requirements.txt 생성 + git push ──────────────────────\n"
    "# pip freeze 결과로 requirements.txt를 교체합니다.\n"
    "import subprocess, sys, os\n"
    "from datetime import datetime\n"
    "from pathlib import Path\n"
    "\n"
    "# pip freeze 실행\n"
    "r = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],\n"
    "                   capture_output=True, text=True)\n"
    "lines = r.stdout.strip().split('\\n')\n"
    "\n"
    "# 헤더 추가\n"
    "header = '\\n'.join([\n"
    "    f'# 생성: {datetime.now().isoformat()}',\n"
    "    f'# 실행 환경: Python {sys.version.split()[0]} / Kaggle Notebook',\n"
    "    '# 과제 지정: Python 3.10.12 / torch 2.5.1',\n"
    "    '',\n"
    "])\n"
    "Path('requirements.txt').write_text(header + r.stdout, encoding='utf-8')\n"
    "\n"
    "# torch 버전 확인\n"
    "torch_line = next((l for l in lines if l.startswith('torch==')), '없음')\n"
    "ok = 'OK  ' if '2.5.1' in torch_line else 'WARN'\n"
    "print(f'{ok} {torch_line}')\n"
    "print(f'requirements.txt 저장 ({len(lines)}개 패키지)')\n"
    "\n"
    "# git commit + push\n"
    "print()\n"
    "print('git push 중...')\n"
    "os.system('git config user.email \"auto@kaggle\" 2>/dev/null')\n"
    "os.system('git config user.name \"Kaggle\" 2>/dev/null')\n"
    "os.system('git add outputs/ requirements.txt')\n"
    "os.system('git commit -m \"feat: Kaggle 실행 결과 업데이트\"')\n"
    "os.system('git push origin main')\n"
    "print()\n"
    "print('완료! GitHub에서 outputs/ 폴더를 확인하세요.')",
    "cell-13",
))

# ── 노트북 조립 ────────────────────────────────────────────────────────────────

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent.parent / "notebooks" / "kaggle_main.ipynb"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"생성 완료: {out_path}")
print(f"셀 수    : {len(cells)}개 (코드 {sum(1 for c in cells if c['cell_type']=='code')}개, 마크다운 {sum(1 for c in cells if c['cell_type']=='markdown')}개)")
