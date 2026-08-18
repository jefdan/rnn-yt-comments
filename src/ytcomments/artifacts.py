from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelMetadata:
    version: int
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    special_tokens: dict[str, int]
    preprocessing_version: str
    training_config: dict[str, object]


@dataclass
class ModelArtifact:
    metadata: ModelMetadata
    vocab_to_int: dict[str, int]
    int_to_vocab: dict[int, str]
    state_dict: dict[str, torch.Tensor]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, directory / "weights.pt")
        (directory / "vocab.json").write_text(
            json.dumps({"vocab_to_int": self.vocab_to_int, "int_to_vocab": {str(k): v for k, v in self.int_to_vocab.items()}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "metadata.json").write_text(json.dumps(asdict(self.metadata), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: torch.device | str = "cpu") -> "ModelArtifact":
        metadata = ModelMetadata(**json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        state_dict = torch.load(directory / "weights.pt", map_location=device, weights_only=True)
        return cls(metadata, vocab["vocab_to_int"], {int(k): v for k, v in vocab["int_to_vocab"].items()}, state_dict)
