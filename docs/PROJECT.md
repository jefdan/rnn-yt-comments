# Project Brief

## Purpose

Build a local-first tool that turns comments from a configured set of YouTube videos into a trainable text-generation model. The same project should support a guided terminal user interface (TUI), direct CLI commands, and an optional FastAPI service.

## Primary Workflow

1. Read video URLs from root-level `links.yaml`.
2. Validate and deduplicate the URLs.
3. Download comment metadata with the bundled or configured `yt-dlp` executable.
4. Normalize comments into a versioned dataset.
5. Train an RNN from the dataset and save a self-describing model artifact.
6. Generate comments interactively from the TUI or CLI.
7. Optionally expose generation and health/status operations through FastAPI.

## Non-Goals For The First Release

- Cloud training or hosted inference.
- A web frontend replacing the TUI.
- Automatic publishing of generated comments to YouTube.
- A promise that generated text is factual, safe, or suitable for posting without review.

## User Experience

The TUI should make the common path discoverable: inspect configuration, download, preprocess, train, generate, and inspect run status. Long-running work must show progress, preserve logs, and fail with actionable messages. CLI subcommands remain available for scripting and automation.

## Success Criteria

- A fresh checkout can be configured with `links.yaml` and run without relying on the current working directory.
- Each pipeline stage has a stable command and a testable Python API.
- Dataset and model outputs are isolated under an explicit workspace directory.
- Inference can load a model without importing training or GUI code.
- FastAPI is optional and does not become a required dependency for the TUI/CLI path.
