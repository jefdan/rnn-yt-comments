import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from model import RNN, device
import random

comments = []
for filename in os.listdir('./comments-json'):
    if filename.endswith('.json'):
        with open(
                os.path.join('./comments-json', filename),
                'r',
                encoding='utf-8') as f:
            data = json.load(f)
            comments.extend(
                [comment['text'] for comment in data.get('comments', [])])

random.shuffle(comments)
max_comments = 10000  # Adjust this number based on your VRAM size.
comments = comments[:max_comments]

all_text = ' '.join(comments)
words = all_text.split()
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
vocab_to_int = {word: idx for idx, word in enumerate(vocab, 3)}
vocab_to_int['<EOS>'] = 0
vocab_to_int['<UNK>'] = 1
vocab_to_int['<START>'] = 2
int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

torch.save(vocab_to_int, 'vocab_to_int.pth')
torch.save(int_to_vocab, 'int_to_vocab.pth')

special_tokens = ['<START>', '<EOS>', '<UNK>']
vocab_size = len(vocab_to_int) - len(special_tokens) + 1

with open('vocab_size.txt', 'w') as f:
    f.write(str(vocab_size))

comments_int = [
    [vocab_to_int.get(
        word,
        vocab_to_int['<UNK>'])
        for word in comment.split()] for comment in comments]


class CommentsDataset(Dataset):
    def __init__(self, comments):
        self.comments = comments

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return torch.tensor(self.comments[idx], dtype=torch.long)


def collate_fn(batch):
    batch = [item for item in batch if len(item) > 1]
    if len(batch) == 0:
        return torch.empty(0, dtype=torch.long)
    return pad_sequence(batch, batch_first=True, padding_value=0)


dataset = CommentsDataset(comments_int)
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

vocab_size = len(vocab) + 1
embedding_dim = 64
hidden_dim = 128
output_dim = vocab_size
learning_rate = 0.001
num_epochs = 10

print(f"Using device: {device}")

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for inputs in dataloader:
        if inputs.size(0) == 0:
            continue

        inputs = inputs.to(device)
        targets = inputs[:, 1:].contiguous().view(-1).to(device)
        inputs = inputs[:, :-1]

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, vocab_size)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

torch.save(model.state_dict(), 'rnn_model.pth')
