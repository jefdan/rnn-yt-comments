from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import ModelArtifact, ModelMetadata
from .config import ProjectConfig
from .dataset import load_comments
from .model import CommentRNN

SPECIAL = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2, "<START>": 3}


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    artifact_dir: Path
    comments: int
    final_loss: float


ProgressCallback = Callable[[str], None]


class CommentDataset(Dataset):
    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.sequences[index], dtype=torch.long)


def _collate(batch: list[torch.Tensor]) -> torch.Tensor:
    longest = max(len(item) for item in batch)
    result = torch.zeros((len(batch), longest), dtype=torch.long)
    for index, item in enumerate(batch):
        result[index, :len(item)] = item
    return result


def train(config: ProjectConfig, *, epochs: int = 3, max_comments: int | None = None, seed: int = 42, progress: ProgressCallback | None = None) -> TrainingResult:
    comments = load_comments(config)
    random.Random(seed).shuffle(comments)
    comments = comments[:max_comments] if max_comments else comments
    if not comments:
        raise ValueError("No comments available for training")
    counts = Counter(word for comment in comments for word in comment.split())
    vocab_to_int = dict(SPECIAL)
    vocab_to_int.update({word: index for index, (word, _) in enumerate(counts.most_common(), start=len(SPECIAL))})
    int_to_vocab = {index: word for word, index in vocab_to_int.items()}
    sequences = [[vocab_to_int["<START>"]] + [vocab_to_int.get(word, SPECIAL["<UNK>"]) for word in comment.split()] + [vocab_to_int["<EOS>"]] for comment in comments]
    dataset = CommentDataset(sequences)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommentRNN(len(vocab_to_int), 64, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    final_loss = 0.0
    batches = len(loader)
    if progress:
        progress(f"Training {len(comments)} comments on {device} ({epochs} epochs, {batches} batches/epoch)")
    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            inputs, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs).reshape(-1, len(vocab_to_int)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            if progress:
                progress(f"Epoch {epoch + 1}/{epochs}, batch {batch_index}/{batches}, loss={final_loss:.4f}")
    run_id = uuid.uuid4().hex[:12]
    metadata = ModelMetadata(1, len(vocab_to_int), 64, 128, SPECIAL, "word-v1", {"epochs": epochs, "seed": seed})
    artifact_dir = config.artifacts_dir / run_id
    ModelArtifact(metadata, vocab_to_int, int_to_vocab, {key: value.detach().cpu() for key, value in model.state_dict().items()}).save(artifact_dir)
    (config.artifacts_dir / "latest").write_text(run_id, encoding="utf-8")
    if progress:
        progress(f"Saved model run {run_id}")
    return TrainingResult(run_id, artifact_dir, len(comments), final_loss)
