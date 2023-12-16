import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Load your custom tokenizer (Replace with your actual loading method)
tokenizer = Tokenizer.from_file("path/to/your/tokenizer.json")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = [tokenizer.encode(text).ids for text in texts]
        self.labels = labels
        self.max_length = max_length

    def __getitem__(self, idx):
        item = {'input_ids': self.encodings[idx][:self.max_length]}
        item['input_ids'] += [0] * (self.max_length - len(item['input_ids']))  # Padding
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# Parameters
VOCAB_SIZE = tokenizer.get_vocab_size()  # Get vocab size from the tokenizer
EMBED_DIM = 100
NUM_CLASSES = 2  # Modify based on your task
MAX_LENGTH = 100  # Maximum length of input text

# Load your dataset (Replace with actual data)
train_texts, train_labels = ..., ...  # Replace with your data
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = torch.tensor(batch['input_ids']).long()
        labels = torch.tensor(batch['labels']).long()

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
