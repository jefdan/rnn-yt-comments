from pathlib import Path

import pytest

from ytcomments.config import load_config


def test_load_config_supports_advanced_video_options(tmp_path: Path) -> None:
    path = tmp_path / "links.yaml"
    path.write_text("""settings:\n  language: en\nvideos:\n  - url: https://youtu.be/example\n    name: demo\n    max_comments: 10\n    options:\n      future_flag: true\n""", encoding="utf-8")
    config = load_config(path)
    assert config.videos[0].name == "demo"
    assert config.videos[0].max_comments == 10
    assert config.videos[0].options["future_flag"] is True


def test_load_config_rejects_non_youtube_urls(tmp_path: Path) -> None:
    path = tmp_path / "links.yaml"
    path.write_text("videos:\n  - url: https://example.com/video\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
