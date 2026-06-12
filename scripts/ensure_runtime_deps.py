"""Install grading-runtime dependencies from requirements.txt."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

PROFILES = {
    "classifier": {
        "transformers": "transformers",
        "datasets": "datasets",
        "scikit-learn": "sklearn",
        "tqdm": "tqdm",
    },
    "chatbot": {
        "transformers": "transformers",
        "tokenizers": "tokenizers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "chromadb": "chromadb",
        "sentence-transformers": "sentence_transformers",
        "rank-bm25": "rank_bm25",
        "kiwipiepy": "kiwipiepy",
        "requests": "requests",
        "beautifulsoup4": "bs4",
        "gradio": "gradio",
    },
}
PROFILES["all"] = PROFILES["classifier"] | PROFILES["chatbot"]
SMOKE_IMPORTS = {
    "classifier": ("transformers", "datasets", "sklearn"),
    "chatbot": ("chromadb", "kiwipiepy", "rank_bm25", "sentence_transformers"),
}
SMOKE_IMPORTS["all"] = SMOKE_IMPORTS["classifier"] + SMOKE_IMPORTS["chatbot"]


def load_requirement_specs() -> dict[str, str]:
    specs: dict[str, str] = {}
    for raw in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        spec = raw.strip()
        if not spec or spec.startswith("#"):
            continue
        package_name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0]
        specs[package_name.casefold()] = spec
    return specs


def profile_specs(profile: str) -> list[str]:
    requirement_specs = load_requirement_specs()
    required_imports = PROFILES[profile]

    undefined = set(required_imports) - set(requirement_specs)
    if undefined:
        raise ValueError(
            f"requirements.txt에 필수 패키지가 없습니다: {sorted(undefined)}"
        )

    return [requirement_specs[name] for name in required_imports]


def verify_imports(profile: str) -> None:
    importlib.invalidate_caches()
    failed = []
    for module in SMOKE_IMPORTS[profile]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed.append(f"{module}: {exc}")

    if failed:
        details = "\n  - ".join(failed)
        raise RuntimeError(f"필수 패키지 import 검증 실패:\n  - {details}")

    print(f"[deps] {profile} import 검증 완료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="all",
        help="설치할 실행 경로의 의존성 범위",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="이미 import 가능한 패키지는 설치 대상에서 제외",
    )
    args = parser.parse_args()

    specs = profile_specs(args.profile)
    if args.missing_only:
        specs = [
            spec
            for spec, module in zip(specs, PROFILES[args.profile].values())
            if importlib.util.find_spec(module) is None
        ]

    if not specs:
        print(f"[deps] {args.profile} 필수 패키지가 이미 설치되어 있습니다.")
        verify_imports(args.profile)
        return

    print(f"[deps] {args.profile} 필수 패키지 설치: {', '.join(specs)}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade-strategy",
            "only-if-needed",
            *specs,
        ]
    )
    print("[deps] 설치 완료")
    verify_imports(args.profile)


if __name__ == "__main__":
    main()
