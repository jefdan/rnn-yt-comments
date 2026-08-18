# YouTube Comment RNN

A local-first pipeline that downloads YouTube comments with yt-dlp, preprocesses them, trains a word-level recurrent neural network, and responds to prompts through a CLI, Textual chat interface, or FastAPI.

## Quick Start

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ytcomments status
ytcomments download
ytcomments preprocess
ytcomments train
ytcomments generate "Write a funny comment about this video"
```

The interactive terminal interface is available with `ytcomments tui` or `ytcomments-tui`. Enter a message and press Enter or Send to receive a response. Long-running TUI actions run in the background and stream progress into the log. CLI download, preprocessing, and training commands also print live progress. The API runs with `ytcomments-api` and exposes `/health`, `/status`, and `POST /generate`.

For API generation, send a JSON body such as `{"prompt":"Write a funny comment about this video","temperature":0.9,"max_length":80}`.

## Configuration

The root-level [links.yaml](links.yaml) is the canonical input. It supports global settings and future per-video options:

```yaml
settings:
  max_comments: null
  language: en
videos:
  - url: https://www.youtube.com/watch?v=example
    name: example
    enabled: true
    max_comments: null
    options: {}
```

Generated data lives under `data/`, model runs under `artifacts/models/`, and run logs/checkpoints belong under `runs/`. Model artifacts contain their vocabulary, special tokens, dimensions, preprocessing version, and training metadata.

## Development

```powershell
pip install -e ".[test]"
python -m pytest
```

The architecture and implementation plan are in [docs/PROJECT.md](docs/PROJECT.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/ROADMAP.md](docs/ROADMAP.md).

Downloading comments can be rate-limited by YouTube and is subject to the relevant platform terms. This is prompt-conditioned comment generation, not an instruction-tuned conversational model; the training data contains standalone comments rather than dialogue pairs. Responses should be reviewed before use.
