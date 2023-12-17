import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Load your custom tokenizer (Replace with your actual loading method)
tokenizer = Tokenizer.from_file("tiny_stories_tokenizer.json")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = [tokenizer.encode(text).ids for text in texts]
        self.labels = labels
        self.max_length = max_length

    def __getitem__(self, idx):
        # Pad the sequence to max_length
        input_ids = self.encodings[idx][:self.max_length] + [0] * (self.max_length - len(self.encodings[idx]))
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

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
train_texts = ["I love this product", "Worst experience ever", "Happy with the purchase", "Not what I expected"]
train_labels = [1, 0, 1, 0]  # Assuming 1 = Positive, 0 = Negative # Replace with your data
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
        #print(batch['input_ids'])  # Check the structure
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long)  # Ensure input_ids is a long tensor
        labels = torch.tensor(batch['labels'], dtype=torch.long)  # Ensure labels is a long tensor

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)  # labels should be a 1D tensor of integer labels

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


