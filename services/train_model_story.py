import torch
import torch.nn as nn
from tokenizers import Tokenizer

# Load your custom tokenizer
tokenizer = Tokenizer.from_file("tiny_stories_tokenizer.json")

class          StoryGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super(StoryGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))

def generate_story(model, start_string, generation_length=100):
    # Convert start string to tokens
    input_eval = tokenizer.encode(start_string).ids
    input_eval = torch.tensor([input_eval], dtype=torch.long)
    
    model.eval()  # Evaluation mode
    text_generated = []

    prev_state = model.init_state(1)

    with torch.no_grad():
        for _ in range(generation_length):
            logits, prev_state = model(input_eval, prev_state)
            probabilities = torch.nn.functional.softmax(logits[:, -1], dim=1)
            predicted_id = torch.multinomial(probabilities, num_samples=1)[-1,0].item()
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long)
            text_generated.append(predicted_id)

    story = tokenizer.decode(text_generated)
    return story

# Model Parameters (Adjust as needed)
VOCAB_SIZE = tokenizer.get_vocab_size()
EMBED_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2

# Initialize the model
model = StoryGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS)

# Load the trained model (ensure that the model is trained and the path is correct)
# model.load_state_dict(torch.load('path/to/your/model.pth'))

# Example usage
prompt = "Once upon a time, there was a"
story = generate_story(model, prompt, generation_length=200)
print(story)
