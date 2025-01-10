import torch
import torch.nn as nn
import numpy as np
from model import RNN, vocab_to_int, int_to_vocab, device

# Load the vocabulary size
with open('vocab_size.txt', 'r') as f:
    vocab_size = int(f.read())

embedding_dim = 64
hidden_dim = 128
output_dim = vocab_size

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('rnn_model.pth', weights_only=True))
model.eval()

# Function to generate new comments
def generate_comment(model, vocab_to_int, int_to_vocab, max_length=100):
    model.eval()
    words = ['<START>']

    for _ in range(max_length):
        input_seq = torch.tensor([[vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in words]], dtype=torch.long).to(device)
        output = model(input_seq)
        last_word_logits = output[0, -1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_idx = np.random.choice(len(last_word_logits), p=p)
        if word_idx in int_to_vocab:
            words.append(int_to_vocab[word_idx])
        else:
            words.append('<UNK>')  # Handle unknown words

        if int_to_vocab.get(word_idx) == '<EOS>':
            break

    return ' '.join(words[1:])  # Exclude the <START> token

# Example usage
start_text = "This video"
generated_comment = generate_comment(model, vocab_to_int, int_to_vocab)
print(generated_comment)
