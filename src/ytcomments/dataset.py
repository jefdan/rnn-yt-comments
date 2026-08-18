from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import ProjectConfig
from .download import raw_files


@dataclass(frozen=True)
class DatasetResult:
    path: Path
    comments: int


ProgressCallback = Callable[[str], None]


def normalize_comment(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_comments(document: object) -> list[str]:
    if not isinstance(document, dict):
        return []
    comments = document.get("comments", [])
    if not isinstance(comments, list):
        return []
    return [normalize_comment(str(item["text"])) for item in comments if isinstance(item, dict) and item.get("text")]


def preprocess(config: ProjectConfig, progress: ProgressCallback | None = None) -> DatasetResult:
    config.ensure_directories()
    output = config.processed_dir / "comments.jsonl"
    seen: set[str] = set()
    count = 0
    with output.open("w", encoding="utf-8") as destination:
        sources = raw_files(config)
        for index, source in enumerate(sources, start=1):
            if progress:
                progress(f"[{index}/{len(sources)}] Reading {source.name}")
            try:
                document = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for comment in _extract_comments(document):
                if comment and comment not in seen:
                    seen.add(comment)
                    destination.write(json.dumps({"text": comment}, ensure_ascii=False) + "\n")
                    count += 1
    if progress:
        progress(f"Processed {count} unique comments")
    return DatasetResult(output, count)


def load_comments(config: ProjectConfig) -> list[str]:
    path = config.processed_dir / "comments.jsonl"
    if not path.exists():
        raise FileNotFoundError("Processed dataset not found; run preprocess first")
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
