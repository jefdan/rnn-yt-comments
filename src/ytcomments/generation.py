from __future__ import annotations

from pathlib import Path

import torch

from .artifacts import ModelArtifact
from .model import CommentRNN


def latest_artifact(artifacts_dir: Path) -> Path:
    pointer = artifacts_dir / "latest"
    if not pointer.exists():
        raise FileNotFoundError("No trained model found; run train first")
    return artifacts_dir / pointer.read_text(encoding="utf-8").strip()


def generate(artifact_dir: Path, *, prompt: str = "", max_length: int = 80, temperature: float = 0.9, seed: int | None = None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = ModelArtifact.load(artifact_dir, device)
    metadata = artifact.metadata
    model = CommentRNN(metadata.vocab_size, metadata.embedding_dim, metadata.hidden_dim).to(device)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    prompt_tokens = prompt.strip().split()
    current = [metadata.special_tokens["<START>"]]
    current.extend(artifact.vocab_to_int.get(token, metadata.special_tokens["<UNK>"]) for token in prompt_tokens)
    generated: list[str] = []
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(torch.tensor([current], dtype=torch.long, device=device))[0, -1] / max(temperature, 0.05)
            token = int(torch.multinomial(torch.softmax(logits, dim=0), 1).item())
            word = artifact.int_to_vocab.get(token, "<UNK>")
            if word == "<EOS>":
                if generated:
                    break
                continue
            if word not in {"<PAD>", "<START>"}:
                generated.append(word)
            current.append(token)
    return " ".join(generated)
