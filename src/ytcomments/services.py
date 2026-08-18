from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig
from .dataset import DatasetResult, preprocess
from .download import DownloadResult, download_comments
from .generation import generate, latest_artifact
from .training import TrainingResult, train


def project_status(config: ProjectConfig) -> dict[str, object]:
    processed = config.processed_dir / "comments.jsonl"
    latest = config.artifacts_dir / "latest"
    return {
        "videos": sum(1 for video in config.videos if video.enabled),
        "raw_files": len(list(config.raw_dir.glob("**/*.json"))),
        "processed": processed.exists(),
        "model": latest.exists(),
    }


class ProjectService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def download(self, progress=None) -> list[DownloadResult]:
        return download_comments(self.config, progress)

    def preprocess(self, progress=None) -> DatasetResult:
        return preprocess(self.config, progress)

    def train(self, **kwargs: object) -> TrainingResult:
        return train(self.config, **kwargs)

    def generate(self, **kwargs: object) -> str:
        return generate(latest_artifact(self.config.artifacts_dir), **kwargs)

    def status(self) -> dict[str, object]:
        return project_status(self.config)
