# Decision Log

## 2026-08-18: YAML Is The Source Of Video Inputs

`links.yaml` at the repository root is the canonical input file. It is structured rather than line-oriented so future options such as labels or per-video limits can be added without another migration.

## 2026-08-18: Services Before Interfaces

Download, preprocessing, training, and inference will be implemented as importable services. CLI, TUI, and FastAPI are adapters over those services, preventing three separate implementations of the workflow.

## 2026-08-18: Self-Describing Model Artifacts

Weights alone are insufficient because the current project stores vocabulary and dimensions in separate files. A model run will package its weights and all required metadata together, with explicit compatibility validation.

## 2026-08-18: Initial Interface Choices

Textual is the TUI framework, FastAPI is included in the default installation, and Python 3.11 is the minimum supported version. `links.yaml` accepts structured video entries with future-facing fields such as `name`, `enabled`, `max_comments`, and `options`.

The first model keeps the project's word-level RNN idea but replaces its loose collection of `.pth` files with a versioned, self-describing artifact. This preserves the project's character while making future model changes explicit.
