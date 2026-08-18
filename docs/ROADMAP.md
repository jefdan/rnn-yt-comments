# Roadmap

## Phase 1: Foundation

- Add a package layout and one application configuration object.
- Replace `links.txt` with validated root-level `links.yaml`.
- Add a console entry point with `status`, `download`, `preprocess`, `train`, and `generate` commands.
- Add focused tests for config parsing, paths, URL handling, and artifact metadata.

## Phase 2: Pipeline

- Implement raw download storage and resumable behavior.
- Normalize yt-dlp comment JSON into a versioned processed dataset.
- Implement a configurable RNN training service with checkpoints and metrics.
- Make generation load one self-contained artifact.

## Phase 3: TUI

- Build a Textual dashboard for pipeline status and recent runs.
- Add interactive controls for download, preprocessing, training, and generation.
- Surface logs and errors without blocking the event loop.

## Phase 4: API And Hardening

- Add an optional FastAPI app with health, model status, and generation endpoints.
- Add API tests and request validation.
- Document installation profiles, GPU/CPU behavior, privacy, rate limits, and recovery from interrupted runs.
- Remove or archive legacy script entry points after parity is verified.

## Validation Gates

Each phase must leave the previous workflow usable. No phase is complete without automated tests for the changed service boundary and a documented manual smoke test.
