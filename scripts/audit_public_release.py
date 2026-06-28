#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Construct private-context patterns from fragments so this public file does not
# itself contain those phrases verbatim.
PRIVATE_PATTERNS = [
    "Sa" + "kana",
    "Technical" + " Problem" + " Set",
    "Part" + " 5",
    "Effort" + "300",
    r"V1\.4",
    "v" + "146",
    "native" + "_holdout",
    "Recruit" + "er",
    "/mnt" + "/ssd",
    "provider" + " console",
    "observed" + "_provider" + "_cost",
]
SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z_-]{20,}",
    r"sk-[A-Za-z0-9_-]{20,}",
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".js", ".tex", ".cff", ".bib"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    hits = []
    patterns = [re.compile(p) for p in PRIVATE_PATTERNS + SECRET_PATTERNS]
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            for pat in patterns:
                if pat.search(line):
                    hits.append((str(path.relative_to(root)), i, pat.pattern, line[:220]))
    if hits:
        print("PUBLIC_AUDIT_HITS")
        for file, line, pat, snippet in hits[:200]:
            print(f"{file}:{line}:/{pat}/ {snippet}")
        raise SystemExit(1)
    print("PUBLIC_AUDIT_OK")


if __name__ == "__main__":
    main()
