import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import json
from datasets import load_dataset
from tqdm import tqdm
import os

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


'''
# this is beam generation
class GenerateStory:
    @staticmethod
    def generate_story(model, prompt, tokenizer, max_length=50, beam_width=1):
        model.eval()

        # Initialize the beam with the start token
        start_token_id = tokenizer.encode(prompt)[0]
        beam = [(torch.tensor([[start_token_id]], dtype=torch.long), 0)]  # List of tuples (sequence, score)

        for step in range(max_length):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == tokenizer.vocab.get('<eos>'):  # End of sequence token
                    candidates.append((seq, score))
                    continue

                output = model(seq, seq)
                probs = torch.softmax(output[:, -1, :], dim=-1)
                top_probs, top_ids = probs.topk(beam_width)

                for i in range(beam_width):
                    next_id = top_ids[0, i].unsqueeze(0).unsqueeze(0)
                    next_score = score + torch.log(top_probs[0, i])
                    next_seq = torch.cat([seq, next_id], dim=1)
                    candidates.append((next_seq, next_score))

                    # Debugging: Print the tokens being added
                    token = tokenizer.decode(next_id[0])
                    print(f"Step {step}, Adding token: '{token}' to sequence")

            # Sort all candidates by score and select the best ones
            ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
            beam = ordered[:beam_width]

            # Enhanced Debugging: Print the sequences in the beam after each step
            for seq, _ in beam:
                sequence_tokens = tokenizer.decode(seq[0])
                print(f"Beam Contents at Step {step}: '{sequence_tokens}'")

            # Log intermediate outputs
            print(f"Step {step}, Beam Contents: {beam}")

        # Select the sequence with the highest score
        best_sequence, _ = beam[0]
        generated_story = tokenizer.decode(best_sequence[0].tolist())
        print(f"Generated story: {generated_story}")
        return generated_story

'''

# this is greedy generation
class GenerateStory:
    @staticmethod
    def generate_story(model, prompt, tokenizer, max_length=100):
        model.eval()

        # Encode the prompt and initialize the sequence
        start_token_ids = tokenizer.encode(prompt)
        seq = torch.tensor([start_token_ids], dtype=torch.long)

        for step in range(max_length):
            if seq.size(1) > max_seq_length:
                seq = seq[:, -max_seq_length:]
            # Check if the last token is the end of sequence token
            if seq[0, -1].item() == tokenizer.vocab.get('<eos>'):
                break

            output = model(seq, seq)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_token_id = probs.argmax(dim=-1).unsqueeze(0)

            # Append the next token ID to the sequence
            seq = torch.cat([seq, next_token_id], dim=1)

            # Debugging: Print the tokens being added
            token = tokenizer.decode(next_token_id[0].tolist())
            print(f"Step {step}, Adding token: '{token}' to sequence")

        generated_story = tokenizer.decode(seq[0].tolist())
        print(f"Generated story: {generated_story}")
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
#num_examples = min(100, len(dataset['train']))
#dataset['train'] = dataset['train'].select(range(num_examples))
#dataset = large_dataset['train']['text'].select(range(25))  # Adjust the range 
torch.autograd.set_detect_anomaly(True) # Utilize PyTorch's debugging tools. This can help identify where NaNs are being introduced.


#print(dataset['train'][0])
tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')
print("tokenizer ready")


'''



# Sample text to test the tokenizer
sample_text = "Hello, this is a test."

# Encode the text
encoded_text = tokenizer.encode(sample_text)
print(f"Encoded Text: {encoded_text}")

# Decode the encoded text back to text
decoded_text = tokenizer.decode(encoded_text)
print(f"Decoded Text: {decoded_text}")

# Check if the decoded text matches the original text
if decoded_text == sample_text:
    print("Tokenizer Test Passed: The decoded text matches the original text.")
else:
    print("Tokenizer Test Failed: The decoded text does not match the original text.")

# Check if the standard space character is in the tokenizer's vocabulary
if " " in tokenizer.vocab:
    space_token_id = tokenizer.vocab[" "]
    print(f"The standard space character is in the vocabulary with token ID: {space_token_id}")
else:
    print("The standard space character is not in the tokenizer's vocabulary.")
'''









tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Add the code here to check the first few samples of tokenized data
#for i in range(5):  # Check first 5 examples
    #print(f"Sample {i}: {tokenized_dataset['train'][i]}")

#for i in range(5):  # Check first 5 examples
  #  print(tokenized_dataset['train'][i])
#print(tokenized_dataset['train'][0]['src'][:10])  # Print first 10 tokens of the first sample

# Split dataset into training and validation
total_size = len(tokenized_dataset['train'])
train_size = int(total_size * 0.8)
val_size = total_size - train_size
from torch.utils.data import random_split

train_dataset, val_dataset = random_split(tokenized_dataset['train'], [train_size, val_size])


# DataLoader
def collate_fn(batch):
    src_batch = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item['tgt'], dtype=torch.long) for item in batch]
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    # Check the shapes after padding
    #print(f"Shape of src_batch after padding: {src_batch.shape}")  # Should be 2D
    #print(f"Shape of tgt_batch after padding: {tgt_batch.shape}")  # Should be 2D

    return src_batch, tgt_batch



#tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')

# Test the tokenizer with individual characters
#for char in ['a', 'b', 'c', ' ']:  # Add more characters to test
    #encoded = tokenizer.encode(char)
#    print(f"Character: {char}, Encoded: {encoded}")

# Test with a word
#encoded = tokenizer.encode("hello")
#print(f"Word: hello, Encoded: {encoded}")





train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)  
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

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
checkpoint_dir = "model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
best_loss = float('inf')
transformer.train()
print("training started")

def check_nan(tensor):
    return torch.isnan(tensor).any()


#from torch.utils.data import DataLoader, random_split

# Assuming you have a dataset object for your entire dataset
# Let's say you want to use 80% of it for training and 20% for validation
#total_size = len(dataset)my_dataset_dictionary['train'][1]
#train_size = int(total_size * 0.8)
#val_size = total_size - train_size

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Now create DataLoaders for training and validation datasets
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)  
#val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)




import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacrebleu.metrics import BLEU

# Assuming you have already defined your Transformer model and CustomTokenizer classes
# ...

# Function to evaluate the model using BLEU score
def evaluate_model(model, val_loader, tokenizer):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            output = model(src_batch, tgt_batch[:, :-1])
            output = torch.argmax(output, dim=-1)
            hypotheses.extend([tokenizer.decode(hyp) for hyp in output])
            references.extend([tokenizer.decode(ref) for ref in tgt_batch[:, 1:]])

    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    return bleu_score.score

createModel = True

if (createModel):

    # Directory for saving model checkpoints
    checkpoint_dir = "model_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_loss = float('inf')

    # Transformer model initialization
    src_vocab_size = tokenizer.get_vocab_size()
    tgt_vocab_size = tokenizer.get_vocab_size()
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
            for src_batch, tgt_batch in train_loader:
                transformer.train()
                optimizer.zero_grad()

                output = transformer(src_batch, tgt_batch[:, :-1])
                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))

                # Check for NaNs in the loss
                if torch.isnan(loss):
                    print("NaN detected in loss")
                    break

                loss.backward()

                # Monitor gradients
                #for name, param in transformer.named_parameters():
                    #if param.grad is not None:
                    # print(f"Gradient of {name}: {param.grad.norm()}")

                #torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0) # Clip gradients, this is optional
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update()

        # Calculate and print average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss}")

        # BLEU Score Evaluation
        bleu_score = evaluate_model(transformer, val_loader, tokenizer)
        #print(f"Epoch {epoch+1} BLEU Score: {bleu_score}")

        # Save checkpoint for each epoch
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
        torch.save(transformer.state_dict(), checkpoint_path)

        # Save the best model based on average loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(transformer.state_dict(), f"{checkpoint_dir}/best_model.pth")


    # Save the final model
    torch.save(transformer, 'transformer_model.pth')
    print("Model saved as transformer_model.pth")
else:
    transformer = torch.load('transformer_model.pth')
    print("Model loaded as transformer_model.pth")

# Load the best model
#transformer.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth"))


# Load your tokenizer
#tokenizer = CustomTokenizer('tiny_stories_tokenizer.json')

# Example usage
prompt = "Once upon a time"
print("prompt ready:" + prompt)
generated_story = GenerateStory.generate_story(transformer, prompt, tokenizer, max_length=100)
print("story ready")
print(generated_story)


