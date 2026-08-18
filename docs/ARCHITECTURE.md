# Target Architecture

## Shape

Use a small Python package with explicit boundaries:

- `src/ytcomments/config.py`: typed settings, paths, and `links.yaml` parsing.
- `src/ytcomments/download.py`: yt-dlp process wrapper, URL validation, progress, and raw-file storage.
- `src/ytcomments/dataset.py`: raw comment readers, normalization, filtering, and dataset manifests.
- `src/ytcomments/model.py`: RNN definition plus serializable model metadata.
- `src/ytcomments/training.py`: vocabulary creation, batching, training, checkpoints, and metrics.
- `src/ytcomments/generation.py`: artifact loading and sampling controls.
- `src/ytcomments/cli.py`: command-line entry points.
- `src/ytcomments/tui.py`: Textual application and screens/actions.
- `src/ytcomments/api.py`: optional FastAPI application factory.

## Data And Artifacts

Keep generated state out of source modules and make it inspectable:

- `data/raw/`: yt-dlp output and download metadata.
- `data/processed/`: normalized comments and a manifest.
- `artifacts/models/<run-id>/`: model weights, vocabulary, configuration, and metrics.
- `runs/<run-id>/`: logs and resumable training state.

A model artifact must include the vocabulary, special-token IDs, model dimensions, preprocessing version, training configuration, and device-independent weights. Loading should use `map_location` and validate metadata before inference.

## Interfaces

The CLI owns orchestration and is the stable automation interface. The TUI calls the same application services rather than shelling out to scripts. The API uses the inference service and read-only status services; it must not duplicate model-loading logic.

## Operational Rules

- Resolve paths from the repository/workspace configuration, never from the process current directory.
- Return structured results from services and translate failures at the UI boundary.
- Use deterministic seeds when requested and record them in run metadata.
- Treat yt-dlp as an external process with captured stdout/stderr and non-zero exit handling.
- Keep FastAPI and Textual dependencies optional only if packaging supports extras cleanly; otherwise document the chosen installation profile.
