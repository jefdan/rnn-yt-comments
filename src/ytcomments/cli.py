from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .services import ProjectService

app = typer.Typer(help="YouTube comment RNN pipeline")


def _service(config: Path) -> ProjectService:
    return ProjectService(load_config(config))


def _progress(message: str) -> None:
    typer.echo(message)


@app.command()
def status(config: Path = typer.Option(Path("links.yaml"), "--config", "-c")) -> None:
    """Show configured videos and available pipeline outputs."""
    typer.echo(_service(config).status())


@app.command()
def download(config: Path = typer.Option(Path("links.yaml"), "--config", "-c")) -> None:
    """Download comments using yt-dlp."""
    for result in _service(config).download(progress=_progress):
        typer.echo(f"{'OK' if result.ok else 'FAILED'} {result.url}: {result.message}")


@app.command()
def preprocess(config: Path = typer.Option(Path("links.yaml"), "--config", "-c")) -> None:
    """Normalize raw comment JSON into the training dataset."""
    result = _service(config).preprocess(progress=_progress)
    typer.echo(f"Wrote {result.comments} comments to {result.path}")


@app.command(name="train")
def train_model(config: Path = typer.Option(Path("links.yaml"), "--config", "-c"), epochs: int = typer.Option(3, min=1), max_comments: int | None = typer.Option(None, min=1)) -> None:
    """Train and save a self-describing model artifact."""
    result = _service(config).train(epochs=epochs, max_comments=max_comments, progress=_progress)
    typer.echo(f"Run {result.run_id}: {result.comments} comments, loss={result.final_loss:.4f}")


@app.command()
def generate(prompt: str = typer.Argument("", help="Message to use as generation context."), config: Path = typer.Option(Path("links.yaml"), "--config", "-c"), temperature: float = typer.Option(0.9, min=0.05, max=3.0), max_length: int = typer.Option(80, min=1, max=500)) -> None:
    """Generate a response to a message using the latest model."""
    typer.echo(_service(config).generate(prompt=prompt, temperature=temperature, max_length=max_length))


@app.command()
def tui(config: Path = typer.Option(Path("links.yaml"), "--config", "-c")) -> None:
    """Open the Textual interface."""
    from .tui import CommentApp
    CommentApp(config).run()


if __name__ == "__main__":
    app()
