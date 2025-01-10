import torch
import torch.nn as nn
import os


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if os.path.exists('vocab_to_int.pth') and os.path.exists('int_to_vocab.pth'):
    vocab_to_int = torch.load('vocab_to_int.pth', weights_only=True)
    int_to_vocab = torch.load('int_to_vocab.pth', weights_only=True)
else:
    vocab_to_int = {}
    int_to_vocab = {}
