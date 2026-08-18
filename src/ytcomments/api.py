from __future__ import annotations

from pathlib import Path

import typer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .services import ProjectService


class GenerateRequest(BaseModel):
    prompt: str = Field("", max_length=2000)
    temperature: float = Field(0.9, ge=0.05, le=3.0)
    max_length: int = Field(80, ge=1, le=500)
    seed: int | None = None


def create_app(config_path: Path | str = "links.yaml") -> FastAPI:
    service = ProjectService(load_config(config_path))
    api = FastAPI(title="YouTube Comment RNN", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/status")
    def status() -> dict[str, object]:
        return service.status()

    @api.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, str]:
        try:
            return {"response": service.generate(**request.model_dump())}
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return api


def serve() -> None:
    import uvicorn
    uvicorn.run("ytcomments.api:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    typer.run(serve)
