from pathlib import Path

from fastapi.testclient import TestClient

from ytcomments.api import create_app


def test_health_endpoint(tmp_path: Path) -> None:
    config = tmp_path / "links.yaml"
    config.write_text("videos:\n  - url: https://youtu.be/example\n", encoding="utf-8")
    client = TestClient(create_app(config))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
