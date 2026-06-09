#!/usr/bin/env python3
"""
Kaggle 실행 스크립트 — NLP Term Project
========================================
사용법:
    python scripts/kaggle_setup_and_run.py --mode check
    python scripts/kaggle_setup_and_run.py --mode install
    python scripts/kaggle_setup_and_run.py --mode refresh
    python scripts/kaggle_setup_and_run.py --mode build_db
    python scripts/kaggle_setup_and_run.py --mode chat
    python scripts/kaggle_setup_and_run.py --mode realtime
    python scripts/kaggle_setup_and_run.py --mode ui
    python scripts/kaggle_setup_and_run.py --mode all

과제 지정 환경: Python 3.10.12 / torch 2.5.1
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── 컬러 출력 (터미널 지원 시) ────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def ok(msg: str):   print(_c(f"[OK  ] {msg}", "92"))
def warn(msg: str): print(_c(f"[WARN] {msg}", "93"))
def err(msg: str):  print(_c(f"[ERR ] {msg}", "91"))
def info(msg: str): print(_c(f"[INFO] {msg}", "96"))
def head(msg: str): print(_c(f"\n{'='*60}\n  {msg}\n{'='*60}", "1;94"))


# ── 프로젝트 루트 자동 감지 ───────────────────────────────────────────────────

def find_root() -> Path:
    """
    다음 경로를 순서대로 탐색해 src/ + data/ 가 있는 곳을 반환.
    없으면 이 스크립트의 상위 디렉토리를 반환.
    """
    candidates = [
        Path("/kaggle/working/NLP_Term_Project"),
        Path("/kaggle/working"),
        Path("/content/NLP_Term_Project"),
        Path("/content/drive/MyDrive/NLP_Term_Project"),
        Path(__file__).resolve().parent.parent,  # scripts/../
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "src").exists() and (p / "data").exists():
            return p.resolve()
    return Path(__file__).resolve().parent.parent


ROOT = find_root()


def setup_env():
    """프로젝트 루트로 이동 + sys.path 등록 + outputs 폴더 생성."""
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    (ROOT / "outputs").mkdir(exist_ok=True)
    info(f"PROJECT ROOT: {ROOT}")


# ── mode: check ───────────────────────────────────────────────────────────────

def mode_check():
    head("환경 확인 (check)")

    # Python 버전
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver_str == "3.10.12":
        ok(f"Python {ver_str}  ← 과제 지정 버전 정확히 일치")
    elif ver.major == 3 and ver.minor == 10:
        warn(f"Python {ver_str}  ← 3.10.x 범위, 마이너 버전 차이 (동작 문제 없음)")
    else:
        warn(f"Python {ver_str}  ← 과제 지정: 3.10.12  (동작 가능하나 버전 다름)")

    # torch 버전
    try:
        import torch
        tv = torch.__version__
        if tv.startswith("2.5.1"):
            ok(f"torch {tv}  ← 과제 지정 버전 일치")
        else:
            warn(f"torch {tv}  ← 과제 지정: 2.5.1  →  --mode install 실행 필요")
        if torch.cuda.is_available():
            ok(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            info(f"CUDA  : {torch.version.cuda}")
        else:
            warn("CUDA 없음 — GPU 모드 불가 (Kaggle Accelerator 설정 확인)")
    except ImportError:
        err("torch 미설치  →  --mode install 실행 필요")

    # 필수 파일 확인
    print()
    info("필수 파일 확인:")
    required = [
        "data/test_cls.json",
        "data/test_chat.json",
        "data/test_realtime.json",
        "data/train.json",
        "data/valid.json",
        "data/processed/chunks.json",
        "src/chatbot_model.py",
        "src/realtime_model.py",
        "src/chatbot_ui.py",
        "classifier_박연진.ipynb",
        "chatbot.sh",
        "scripts/refresh_data.py",
    ]
    missing = []
    for f in required:
        p = ROOT / f
        if p.exists():
            if f.endswith(".json"):
                try:
                    with open(p, encoding="utf-8") as fp:
                        d = json.load(fp)
                    n = len(d) if isinstance(d, list) else "dict"
                    ok(f"  {f}  ({n}건)")
                except Exception as e:
                    warn(f"  {f}  (JSON 오류: {e})")
            else:
                ok(f"  {f}")
        else:
            err(f"  {f}  ← 없음")
            missing.append(f)

    print()
    if missing:
        warn(f"누락 파일 {len(missing)}개 — git pull 또는 데이터 확인 필요")
    else:
        ok("모든 필수 파일 존재")


# ── mode: install ─────────────────────────────────────────────────────────────

def mode_install():
    head("패키지 설치 (install)")

    # 현재 torch 확인
    try:
        import torch
        tv = torch.__version__
    except ImportError:
        tv = "not_installed"

    # torch 2.5.1 설치
    if tv.startswith("2.5.1"):
        ok(f"torch {tv} 이미 설치됨")
    else:
        warn(f"torch {tv} → 2.5.1 설치 시작")
        cuda_tag = "cu121"  # Kaggle 기본 CUDA 12.1
        info(f"wheel: torch==2.5.1+{cuda_tag}")
        r = subprocess.run([
            sys.executable, "-m", "pip", "install",
            f"torch==2.5.1+{cuda_tag}",
            f"torchvision==0.20.1+{cuda_tag}",
            f"torchaudio==2.5.1+{cuda_tag}",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
            "--quiet",
        ])
        if r.returncode == 0:
            ok("torch 2.5.1 설치 완료")
            warn("커널/프로세스 재시작 후 나머지 패키지 설치 필요")
        else:
            err("torch 설치 실패 — CUDA 버전 확인 후 재시도")
            return

    # 나머지 패키지
    print()
    info("추가 패키지 설치:")
    packages = [
        "transformers==4.44.2", "accelerate==0.34.2", "bitsandbytes==0.43.3",
        "peft==0.12.0", "datasets==2.21.0", "trl==0.9.6",
        "sentence-transformers==3.0.1", "chromadb==0.5.5",
        "langchain==0.2.16", "langchain-community==0.2.17",
        "rank-bm25==0.2.2", "kiwipiepy==0.17.1",
        "beautifulsoup4==4.12.3", "requests==2.32.3",
        "lxml==5.3.0", "pdfplumber==0.11.4",
        "scikit-learn==1.5.1", "gradio==4.42.0",
        "tqdm==4.66.5", "python-dotenv==1.0.1",
    ]
    for pkg in packages:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            capture_output=True,
        )
        if r.returncode == 0:
            ok(f"  {pkg}")
        else:
            err(f"  {pkg}  ← 설치 실패")

    # torch 보호 확인
    print()
    try:
        import importlib
        import torch as _t
        importlib.invalidate_caches()
        info(f"torch 현재 버전: {_t.__version__}")
        if not _t.__version__.startswith("2.5.1"):
            warn("torch 버전이 변경됨 — --mode install 재실행 필요")
    except Exception:
        pass


# ── mode: refresh ─────────────────────────────────────────────────────────────

def mode_refresh():
    head("실시간 데이터 갱신 (refresh)")
    info("식단 / 셔틀버스 / 학사일정 / 공지사항 크롤링...")
    r = subprocess.run([sys.executable, "scripts/refresh_data.py"])
    if r.returncode == 0:
        ok("refresh_data 완료")
    else:
        warn(f"refresh_data 일부 실패 (exit {r.returncode}) — 기존 데이터로 진행 가능")


# ── mode: build_db ────────────────────────────────────────────────────────────

def mode_build_db():
    head("ChromaDB 구축 (build_db)")
    chroma_path = ROOT / "chroma_db"
    files = list(chroma_path.iterdir()) if chroma_path.exists() else []

    if files:
        info(f"chroma_db 존재 ({len(files)}개 파일)")
        ans = input("재구축하시겠습니까? [y/N]: ").strip().lower()
        if ans != "y":
            info("건너뜀")
            return
        import shutil
        shutil.rmtree(chroma_path)
        ok("기존 chroma_db 삭제")

    info("inject_faq.py --reset 실행...")
    subprocess.run([sys.executable, "scripts/inject_faq.py", "--reset"])

    info("build_db.py 실행 (2~5분)...")
    r = subprocess.run([sys.executable, "src/vectordb/build_db.py"])
    if r.returncode == 0:
        files_after = list(chroma_path.iterdir()) if chroma_path.exists() else []
        ok(f"ChromaDB 구축 완료 ({len(files_after)}개 파일)")
    else:
        err("ChromaDB 구축 실패")


# ── mode: chat ────────────────────────────────────────────────────────────────

def mode_chat():
    head("Task 2: chat_output.json 생성 (chat)")
    out_path = ROOT / "outputs" / "chat_output.json"
    r = subprocess.run([
        sys.executable, "src/chatbot_model.py",
        "--input",  "data/test_chat.json",
        "--output", "outputs/chat_output.json",
    ])
    if r.returncode == 0 and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        mock = sum(1 for d in data if "[MOCK]" in str(d.get("model", "")))
        if mock:
            warn(f"chat_output.json: {len(data)}건  (MOCK {mock}건 — ChromaDB 확인)")
        else:
            ok(f"chat_output.json: {len(data)}건  (MOCK 없음)")
        print()
        for item in data[:2]:
            q = item.get("user", item.get("question", ""))[:60]
            a = item.get("model", "")[:80]
            print(f"  Q: {q}")
            print(f"  A: {a}")
            print()
    else:
        err(f"chat_output.json 생성 실패 (exit {r.returncode})")


# ── mode: realtime ────────────────────────────────────────────────────────────

def mode_realtime():
    head("Optional: realtime_output.json 생성 (realtime)")
    out_path = ROOT / "outputs" / "realtime_output.json"
    r = subprocess.run([
        sys.executable, "src/realtime_model.py",
        "--input",  "data/test_realtime.json",
        "--output", "outputs/realtime_output.json",
    ])
    if r.returncode == 0 and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        ok(f"realtime_output.json: {len(data)}건")
        for item in data:
            print(f"  Q: {item.get('user', '')[:60]}")
            print(f"  A: {item.get('model', '')[:80]}")
            print()
    else:
        err(f"realtime_output.json 생성 실패 (exit {r.returncode})")


# ── mode: ui ──────────────────────────────────────────────────────────────────

def mode_ui():
    head("Gradio UI 실행 (ui)")
    info("share=True 공개 URL 생성 중... (Ctrl+C로 종료)")
    info("공개 URL(gradio.live)이 출력되면 접속하세요.")
    subprocess.run([sys.executable, "src/chatbot_ui.py"])


# ── output 검증 (all 내부 사용) ───────────────────────────────────────────────

def validate_outputs() -> bool:
    head("Output 파일 검증")
    outputs = [
        ("outputs/chat_output.json",     "Task 2",   True),
        ("outputs/realtime_output.json", "Optional", True),
        ("outputs/cls_output.json",      "Task 1",   False),
    ]
    all_ok = True
    for path, label, check_mock in outputs:
        p = ROOT / path
        if not p.exists():
            err(f"[{label:8}] {path}  ← 없음")
            all_ok = False
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data)
        if check_mock:
            mock = sum(1 for d in data if "[MOCK]" in str(d.get("model", "")))
            if mock:
                warn(f"[{label:8}] {path}: {n}건  (MOCK {mock}건)")
            else:
                ok(f"[{label:8}] {path}: {n}건")
        else:
            ok(f"[{label:8}] {path}: {n}건")
    return all_ok


# ── requirements.txt 생성 (all 내부 사용) ────────────────────────────────────

def gen_requirements():
    head("requirements.txt 생성")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True,
    )
    lines = r.stdout.strip().split("\n")

    header = "\n".join([
        f"# 생성: {datetime.now().isoformat()}",
        f"# 실행 환경: Python {sys.version.split()[0]} / Kaggle Notebook",
        "# 과제 지정: Python 3.10.12 / torch 2.5.1",
        "",
    ])
    req_path = ROOT / "requirements.txt"
    req_path.write_text(header + r.stdout, encoding="utf-8")

    torch_line = next((l for l in lines if l.startswith("torch==")), None)
    if torch_line:
        if "2.5.1" in torch_line:
            ok(torch_line)
        else:
            warn(f"{torch_line}  ← 과제 지정: torch==2.5.1")
    ok(f"requirements.txt 저장 ({len(lines)}개 패키지)")


# ── mode: all ─────────────────────────────────────────────────────────────────

def mode_all():
    setup_env()
    mode_check()
    mode_refresh()
    mode_build_db()
    mode_chat()
    mode_realtime()
    validate_outputs()
    gen_requirements()


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle 실행 스크립트 — NLP Term Project",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["check", "install", "refresh", "build_db", "chat", "realtime", "ui", "all"],
        help=(
            "check    : Python/torch/GPU/파일 환경 확인\n"
            "install  : torch 2.5.1 + 필수 패키지 설치\n"
            "refresh  : 실시간 데이터 갱신 (식단·셔틀·학사일정)\n"
            "build_db : ChromaDB 구축\n"
            "chat     : Task 2  →  outputs/chat_output.json\n"
            "realtime : Optional  →  outputs/realtime_output.json\n"
            "ui       : Gradio UI 실행 (공개 URL 출력)\n"
            "all      : refresh → build_db → chat → realtime → 검증\n"
        ),
    )
    args = parser.parse_args()
    setup_env()

    dispatch = {
        "check":    mode_check,
        "install":  mode_install,
        "refresh":  mode_refresh,
        "build_db": mode_build_db,
        "chat":     mode_chat,
        "realtime": mode_realtime,
        "ui":       mode_ui,
        "all":      mode_all,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
