from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from .config import load_config
from .services import ProjectService


class CommentApp(App[None]):
    CSS = """
    Screen { background: $surface; }
    #main { padding: 1 2; }
    #status { height: 3; border: round $accent; padding: 1; }
    #chat { height: 3; margin-top: 1; }
    #prompt { width: 1fr; }
    #log { height: 1fr; border: round $primary; margin-top: 1; }
    Button { margin-right: 1; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, config_path: Path | str = "links.yaml") -> None:
        super().__init__()
        self.service = ProjectService(load_config(config_path))

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Static("YouTube Comment Chat", id="title")
            yield Label("Loading project status...", id="status")
            with Horizontal():
                yield Button("Download", id="download")
                yield Button("Preprocess", id="preprocess")
                yield Button("Train", id="train")
            with Horizontal(id="chat"):
                yield Input(placeholder="Message the model...", id="prompt")
                yield Button("Send", id="send", variant="primary")
            yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        self.query_one("#status", Label).update(str(self.service.status()))

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)
        self._refresh_status()

    def _set_busy(self, busy: bool) -> None:
        for button in self.query(Button):
            button.disabled = busy

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if prompt:
            self._submit_chat(prompt)

    def _submit_chat(self, prompt: str) -> None:
        self.query_one("#log", RichLog).write(f"[bold cyan]You:[/] {escape(prompt)}")
        self.query_one("#prompt", Input).value = ""
        self._set_busy(True)
        self._run_action("chat", prompt)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            prompt = self.query_one("#prompt", Input).value.strip()
            if prompt:
                self._submit_chat(prompt)
            return
        self._set_busy(True)
        self._run_action(event.button.id or "", "")

    @work(thread=True)
    def _run_action(self, action: str, prompt: str) -> None:
        def progress(message: str) -> None:
            self.call_from_thread(self._log, message)

        try:
            if action == "download":
                for result in self.service.download(progress=progress):
                    progress(f"{'[green]OK[/]' if result.ok else '[red]FAILED[/]'} {result.url}: {result.message}")
            elif action == "preprocess":
                result = self.service.preprocess(progress=progress)
                progress(f"Processed {result.comments} comments")
            elif action == "train":
                result = self.service.train(progress=progress)
                progress(f"Trained {result.run_id}; loss={result.final_loss:.4f}")
            elif action == "chat":
                response = self.service.generate(prompt=prompt)
                progress(f"[bold green]Assistant:[/] {escape(response)}")
        except Exception as error:
            progress(f"[red]{error}[/]")
        finally:
            self.call_from_thread(self._set_busy, False)


def run() -> None:
    CommentApp().run()
