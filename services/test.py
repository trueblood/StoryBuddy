import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import json
from datasets import load_dataset
from tqdm import tqdm

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
class CustomTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as file:
            data = json.load(file)
            # Load the main vocabulary, not just the added_tokens
            self.vocab = data['model']['vocab']
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        # Adjust this method depending on whether you want word-level or character-level tokenization
        # For character-level:
        return [self.vocab.get(char, self.vocab.get('[UNK]')) for char in text]

    def decode(self, token_ids):
        return ''.join(self.inverse_vocab.get(id, '[UNK]') for id in token_ids)

    def get_vocab_size(self):
        return len(self.vocab)

'''
class GenerateStory:
    def generate_story(model, prompt, tokenizer, max_length=50):
        model.eval()
        
        # Encode the prompt
        src_input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        # Initialize the target sequence with a start token or dummy token
        # Assuming your tokenizer has a start token with a specific ID, e.g., <sos_id>
        sos_id = tokenizer.vocab.get('<sos>', 0)  # Replace '<sos>' with your actual start token
        tgt_input_ids = torch.tensor([[sos_id]], dtype=torch.long)

        generated_ids = []

        with torch.no_grad():
            for _ in range(max_length):
                output = model(src_input_ids, tgt_input_ids)
                next_token_id = output[0, -1].argmax().item()
                generated_ids.append(next_token_id)

                # Update the target sequence
                tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([[next_token_id]], dtype=torch.long)], dim=-1)

                # Stop if the end of sentence token is generated
                if next_token_id == tokenizer.vocab.get('<eos>'):
                    break

        generated_story = tokenizer.decode(generated_ids)
        return generated_story
'''    

class GenerateStory:
    def generate_story(model, prompt, tokenizer, max_length=50):
        model.eval()
        
        # Start with just the first token of the encoded prompt
        start_token_id = tokenizer.encode(prompt)[0]
        tgt_input_ids = torch.tensor([[start_token_id]], dtype=torch.long)

        generated_ids = [start_token_id]

        with torch.no_grad():
            for _ in range(max_length):
                output = model(tgt_input_ids, tgt_input_ids)
                next_token_id = output[0, -1].argmax().item()

                # Break if the model starts repeating the same token or generates a [PAD] token
                if next_token_id == generated_ids[-1] or next_token_id == tokenizer.vocab.get('[PAD]', -1):
                    break

                generated_ids.append(next_token_id)
                tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([[next_token_id]], dtype=torch.long)], dim=-1)

        generated_story = tokenizer.decode(generated_ids)
        return generated_story


# Load and preprocess dataset
def tokenize_function(examples, max_seq_length=100):
    # Process each example individually
    #src_encoded = [tokenizer.encode(text) for text in examples['text']]
    #tgt_encoded = [tokenizer.encode(text) for text in examples['text']]
    
   # return {
   #     'src': src_encoded,
   #     'tgt': tgt_encoded
   # }
    src_encoded = []
    tgt_encoded = []
    for text in examples['text']:
        # Tokenize, truncate, and pad
        encoded = tokenizer.encode(text)[:max_seq_length]
        encoded += [0] * (max_seq_length - len(encoded))  # Padding
        src_encoded.append(encoded)
        tgt_encoded.append(encoded)  # Assuming src and tgt are the same

    return {
        'src': src_encoded,
        'tgt': tgt_encoded
    }

dataset = load_dataset("roneneldan/TinyStories")
#dataset = org_dataset['train'].select(range(100)) 

#print(dataset);
# Select a smaller subset of the dataset
num_examples = min(100, len(dataset['train']))
dataset['train'] = dataset['train'].select(range(num_examples))
#dataset = large_dataset['train']['text'].select(range(25))  # Adjust the range 
torch.autograd.set_detect_anomaly(True) # Utilize PyTorch's debugging tools. This can help identify where NaNs are being introduced.


#print(dataset['train'][0])
tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')
print("tokenizer ready")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Add the code here to check the first few samples of tokenized data
for i in range(5):  # Check first 5 examples
    print(f"Sample {i}: {tokenized_dataset['train'][i]}")

#for i in range(5):  # Check first 5 examples
  #  print(tokenized_dataset['train'][i])
#print(tokenized_dataset['train'][0]['src'][:10])  # Print first 10 tokens of the first sample


# DataLoader
def collate_fn(batch):
    src_batch = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item['tgt'], dtype=torch.long) for item in batch]
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    # Check the shapes after padding
    print(f"Shape of src_batch after padding: {src_batch.shape}")  # Should be 2D
    print(f"Shape of tgt_batch after padding: {tgt_batch.shape}")  # Should be 2D

    return src_batch, tgt_batch



#tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')

# Test the tokenizer with individual characters
#for char in ['a', 'b', 'c', ' ']:  # Add more characters to test
    #encoded = tokenizer.encode(char)
#    print(f"Character: {char}, Encoded: {encoded}")

# Test with a word
#encoded = tokenizer.encode("hello")
#print(f"Word: hello, Encoded: {encoded}")





train_loader = data.DataLoader(tokenized_dataset['train'], batch_size=64, shuffle=True, collate_fn=collate_fn)  
print("train_loader ready")

src_vocab_size = tokenizer.get_vocab_size()
tgt_vocab_size = tokenizer.get_vocab_size()
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

torch.autograd.set_detect_anomaly(True)
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
print("transformer ready")
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#optimizer = optim.Adam(transformer.parameters(), lr=0.00001)  # Lower learning rate, to help fix issues with the training

transformer.train()
print("training started")

def check_nan(tensor):
    return torch.isnan(tensor).any()

'''
# we like this one
for epoch in range(1):
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for src_batch, tgt_batch in train_loader:
            print(f"src_batch shape: {src_batch.shape}, tgt_batch shape: {tgt_batch.shape}")
            # small check to make rule out data issues
            if check_nan(src_batch) or check_nan(tgt_batch):
                print("NaN value detected in the batch.")
                break
            optimizer.zero_grad()
            output = transformer(src_batch, tgt_batch[:, :-1])
            if torch.isnan(output).any() or torch.isnan(tgt_batch[:, 1:]).any():
                print("NaN detected in model output or target batch")
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
            print(loss)
            if not torch.isnan(loss):
                loss.backward()
                for name, param in transformer.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in parameter: {name}")
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0) #This is optional - Gradient clipping can prevent gradients from becoming too large, which is another potential cause of NaN loss:
            else:
                print("NaN loss detected")
                break
            print(f"Loss: {loss.item()}")
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    print(f"Epoch {epoch+1} completed, Loss: {loss.item()}")  
'''





for epoch in range(10):  # Adjust the number of epochs as needed
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for src_batch, tgt_batch in train_loader:
            # Check for NaN in input tensors
            if torch.isnan(src_batch).any() or torch.isnan(tgt_batch).any():
                print("NaN detected in input tensors")
                break
            #print(src_batch)
            #print(tgt_batch)
            optimizer.zero_grad()
            print()
            # Forward pass through the model
            output = transformer(src_batch, tgt_batch[:, :-1])

            # Check for NaN in model output
            if torch.isnan(output).any():
                print("NaN detected in model output")
                break

            # Compute loss
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))

            # Check loss value before proceeding
            if torch.isnan(loss).any():
                print("NaN loss detected, idk y")
                print(loss)
                break

            print(f"Loss before backward: {loss.item()}")

            # Backward pass and optimization
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

            optimizer.step()

            # Update progress bar
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    print(f"Epoch {epoch+1} completed, Loss: {loss.item()}")


# Load your tokenizer
#tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')

# Example usage
prompt = "Once upon a time"
print("prompt ready:" + prompt)
generated_story = GenerateStory.generate_story(transformer, prompt, tokenizer, max_length=100)
print("story ready")
print(generated_story)
