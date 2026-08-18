import json
from pathlib import Path

from ytcomments.config import load_config
from ytcomments.dataset import preprocess


def test_preprocess_normalizes_and_deduplicates(tmp_path: Path) -> None:
    (tmp_path / "links.yaml").write_text("videos:\n  - url: https://youtu.be/example\n", encoding="utf-8")
    config = load_config(tmp_path / "links.yaml")
    config.raw_dir.mkdir(parents=True)
    (config.raw_dir / "video.json").write_text(json.dumps({"comments": [{"text": " hello   world "}, {"text": "hello world"}]}), encoding="utf-8")
    result = preprocess(config)
    assert result.comments == 1
    assert result.path.read_text(encoding="utf-8").count("hello world") == 1
