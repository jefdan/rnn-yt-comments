from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import ProjectConfig


@dataclass(frozen=True)
class DownloadResult:
    url: str
    ok: bool
    message: str


ProgressCallback = Callable[[str], None]


def download_comments(config: ProjectConfig, progress: ProgressCallback | None = None) -> list[DownloadResult]:
    config.ensure_directories()
    results: list[DownloadResult] = []
    enabled_sources = [source for source in config.videos if source.enabled]
    for index, source in enumerate(enabled_sources, start=1):
        command = [
            str(config.yt_dlp_path), source.url,
            "--output", str(config.raw_dir / "%(id)s"),
            "--write-comments", "--skip-download", "--no-write-thumbnail",
        ]
        if progress:
            progress(f"[{index}/{len(enabled_sources)}] Downloading {source.url}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            output: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                message = line.strip()
                if message:
                    output.append(message)
                    if progress:
                        progress(f"[{index}/{len(enabled_sources)}] {message}")
            return_code = process.wait()
        except OSError as error:
            results.append(DownloadResult(source.url, False, str(error)))
            if progress:
                progress(f"[{index}/{len(enabled_sources)}] Failed: {error}")
            continue
        message = output[-1] if output else "completed"
        ok = return_code == 0
        results.append(DownloadResult(source.url, ok, message))
        if progress:
            progress(f"[{index}/{len(enabled_sources)}] {'Completed' if ok else 'Failed'}: {message}")
    return results


def raw_files(config: ProjectConfig) -> list[Path]:
    return sorted(config.raw_dir.glob("**/*.json"))
