from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class VideoSource:
    url: str
    name: str | None = None
    enabled: bool = True
    max_comments: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    max_comments: int | None = None
    language: str = "en"


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    settings: Settings
    videos: tuple[VideoSource, ...]

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def yt_dlp_path(self) -> Path:
        local = self.root / "yt-dlp" / "yt-dlp.exe"
        return local if local.exists() else self.root / "yt-dlp" / "yt-dlp"

    def ensure_directories(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.artifacts_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        raise ValueError(f"Unsupported YouTube URL: {url}")
    return url


def load_config(path: Path | str = "links.yaml") -> ProjectConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_settings = document.get("settings", {}) or {}
    settings = Settings(
        max_comments=raw_settings.get("max_comments"),
        language=str(raw_settings.get("language", "en")),
    )
    sources: list[VideoSource] = []
    seen: set[str] = set()
    for item in document.get("videos", []) or []:
        if isinstance(item, str):
            item = {"url": item}
        url = _validate_url(str(item["url"]).strip())
        if url in seen:
            continue
        seen.add(url)
        sources.append(VideoSource(
            url=url,
            name=item.get("name"),
            enabled=bool(item.get("enabled", True)),
            max_comments=item.get("max_comments"),
            options=dict(item.get("options", {}) or {}),
        ))
    if not sources:
        raise ValueError("links.yaml must contain at least one video")
    return ProjectConfig(config_path.parent, settings, tuple(sources))
