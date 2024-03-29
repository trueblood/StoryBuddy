import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time 
from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import seaborn
#searborn.set_context(context="talk")
#%matplotlib inline
import torch
import json
from datasets import load_dataset
#from transformers import PreTrainedTokenizerFast
import psutil
import random
from sklearn.model_selection import KFold
import os
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
import pyamdgpuinfo
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re


class EncoderDecorder(nn.Module):
    # A standard Encoder-Decoder architecture. Base for this and many other models.
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecorder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        #Take in and process masked src and target sequences.
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    # Define standard linear + softmax generation step.
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Clones(nn.Module):
    def clones(module, N):
        # Produce N identical layers.
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class Encoder(nn.Module):
    # Core encoder is a stack of N layers
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = Clones.clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    # Construct a layernorm module. Applies dropout, to prevent neural networks from overfitting.
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2
    
class SublayerConnection(nn.Module):
    # A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last.
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Apply residual connection to any sublayer with the same size.
        return x+self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    # Encoder is made up of two layers: multi-head self-attn and position-wise feed forward network
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Clones.clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.dropout = nn.Dropout(dropout)  # Add dropout

    def forward(self, x, mask):
        # Follow Figure 1 (left) for connections.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.dropout(x)  # Apply dropout
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    # Generic N layer decoder with masking.
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = Clones.clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    # Decoder is made of self-attn, src-attn, and feed forward.
    def __init__(self, size, self_attn, src_attn, feed_foward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_foward = feed_foward
        self.sublayer = Clones.clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Follow Figure 1 (right) for connections.
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_foward)
    
    def subsequent_mask(size):
        # Mask out subsequent positions. 
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        return torch.from_numpy(subsequent_mask)==0
        #attn_shape = (1, size, size)
        #subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
        #return subsequent_mask == 0

class Attention(nn.Module):
    def attention(query, key, value, mask=None, dropout=None):
        # Compute 'Scaled Dot Product Attention'
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # Take in model size and number of heads.
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k.
        self.d_k = d_model//h
        self.h = h
        self.linears = Clones.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Implements Figure 2.
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model=>h*d_k.
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = Attention.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    # Implements FFN aka Feed-Forward Network equation.
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    # Use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.
    def __init__(self, d_model, vocab):
        # Embedding layer.
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # Positional Encoding module injects some information about the relative or absolute position of the tokens in the sequence.
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*-(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x+Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    

class MakeModel():
    # Helper: Construct a model from hyperparameters.
    #def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    def make_model(src_vocab, tgt_vocab, N=8, d_model=768, d_ff=4096, h=10, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecorder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), nn.Sequential(Embeddings(d_model, src_vocab), c(position)), nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), Generator(d_model, tgt_vocab))
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        return model
    
class Batch():
    # Object for holding a batch of data with mask during training.
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src!=pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y!=pad).data.sum()
            
    @staticmethod
    def make_std_mask(tgt, pad):
        # Create a mask to hide padding and future words.
        tgt_mask = (tgt!=pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(DecoderLayer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
class TrainModel():
    def run_epoch(data_iter, model, loss_compute, optimizer, epoch, fold):  
        # Standard Training and Logging Function
        # Generate a training and scoring function to keep track of loss.
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            #Helper.print_memory_usage_gpu()
            #print(i)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 1000 == 0 and i != 0:
                save_checkpoint(model, optimizer, epoch, fold, i)
                print("Checkpoint Created: ", i)
            if i%50==1:
                elapsed = time.time()-start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss/batch.ntokens, tokens/elapsed))
                start = time.time()
                tokens = 0
        return total_loss/total_tokens

    def batch_size_fn(new, count, sofar):
        # Keep augmenting batch and calculate total number of tokens + padding.
        global max_src_in_batch, max_tgt_in_batch
        if count==1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, len(new.src))
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg)+2)
        src_elements = count*max_src_in_batch
        tgt_elements = count*max_tgt_in_batch
        return max(src_elements, tgt_elements)
    '''
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        # Greedy decode function.
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            #out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            out = model.decode(memory, src_mask, Variable(ys), Variable(DecoderLayer.subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return 
    '''

    def greedy_decode(model, src, src_mask, max_len, start_symbol, k):
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            out = model.decode(memory, src_mask, Variable(ys), Variable(DecoderLayer.subsequent_mask(ys.size(1)).type_as(src.data)))
            logits = model.generator(out[:, -1])
            probs = F.softmax(logits, dim=-1)
            
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
            next_word_index = topk_indices[0][random.choice(range(k))]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_index)], dim=1)
        
        return ys



class NoamOpt():
    # This is the optimizer.
    # Optim wrapper that implements rate.
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # Update parameters and rate.
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # Implement `lrate` above.
        if step is None:
            step = self._step
        return self.factor*(self.model_size**(-0.5)*min(step**(-0.5), step*self.warmup**(-1.5)))
    
    def get_std_opt(model):
        # Here, factor, model_size, and warmup_steps are parameters that influence the learning rate. 
        '''Steps to Increase Learning Rate Gradually:
        Adjust factor: The factor parameter in NoamOpt multiplies the learning rate. By increasing this factor, you can increase the overall learning rate.
        Change warmup_steps: This parameter controls how long the learning rate will increase before it starts decaying. Reducing the number of warmup_steps will cause the learning rate to increase more rapidly.
        Modify the NoamOpt class or its initialization: If you want more control over the learning rate changes, consider modifying the NoamOpt class or how it's initialized.'''
        factor = 1  # Try increasing this factor, e.g., 2, 3, etc.
        warmup = 4000  # Adjust the warmup steps if needed
        return NoamOpt(model.src_embed[0].d_model, factor, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        #return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

class LabelSmoothing(nn.Module):
    # Implement label smoothing.
    # We employee label smoothing. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0-smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1)==self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data==self.padding_idx)
        if mask.dim()>0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    

class SimpleLossCompute():
    # A simple loss compute and train function.
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm, is_train=True):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))/norm
        if is_train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        return loss.data.item()*norm.float()
    
'''
class MyIterator(data.Iterator):
    # MyIterator class to make sure the 'pad' token is at the end of the sentence.
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size*100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
'''

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
    
    def tokenize_fn(examples):
        return {
            'src': [tokenizer.encode(sentence) for sentence in examples['text']],  # Replace 'source_column_name' with actual column name
            #'tgt': [tokenizer.encode(sentence) for sentence in examples['target_column_name']]   # Replace 'target_column_name' with actual column name
        }
    
class Helper():
    def get_device():
        # Check whether GPU is available and use it if yes.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        print(f"Using device: {device}")
        return device
    '''
    #This function is crucial for converting your tokenized dataset into a format that can be processed by the TrainModel.run_epoch function.
    def data_generator(tokenized_dataset, batch_size, device):
        # Function to yield batches of data
        for i in range(0, len(tokenized_dataset), batch_size):
            # Extract a batch of tokenized data
            batch_data = tokenized_dataset[i:i + batch_size]

            # Debugging print statement
            print("Batch data example:", batch_data[0])

            # Extracting src and assuming trg is the same as src for this example
            src_batch = [torch.tensor(item['src']) for item in batch_data]
            trg_batch = [torch.tensor(item['src']) for item in batch_data]  # Assuming target is same as source

            # Padding the sequences and converting to tensors
            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
            trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

            yield Batch(src_batch, trg_batch, 0)  # 0 is the padding index
            '''

    @staticmethod
    def data_generator(tokenized_dataset, batch_size, device):
        batch = []
        for item in tokenized_dataset:
            batch.append(item)
            if len(batch) == batch_size:
                src_batch = [torch.tensor(d['encoded_text']) for d in batch]
                trg_batch = [torch.tensor(d['encoded_text']) for d in batch]

                src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
                trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

                yield Batch(src_batch, trg_batch, 0)
                batch = []

        if batch:
            src_batch = [torch.tensor(d['encoded_text']) for d in batch]
            trg_batch = [torch.tensor(d['encoded_text']) for d in batch]

            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
            trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

            yield Batch(src_batch, trg_batch, 0)

    def print_memory_usage():
        print(f"Current memory usage: {psutil.virtual_memory().percent}%")

    def print_memory_usage_gpu():
        first_gpu = pyamdgpuinfo.get_gpu(0) # returns a GPUInfo object
        vram_usage = first_gpu.query_vram_usage()
        vram_usage_in_gb = vram_usage / (1024 ** 3)
        gpu_temp_fahrenheit = first_gpu.query_temperature() * 9/5 + 32
        gpu_load = first_gpu.query_load()
        gpu_power = first_gpu.query_power()
        print(f"Current GPU memory usage: {vram_usage_in_gb} GB")
        print(f"Current GPU Load: {gpu_load}")
        print(f"Current GPU Power: {gpu_power} W")
        print(f"Current GPU Temp: {gpu_temp_fahrenheit} F")

    '''def print_number_epochs(batchSize, tokenized_dataset):
        # this lets me know how many loops that will run
        total_examples = len(tokenized_dataset['train'])  # Total number of examples in the dataset
        batch_size = batchSize 

        # Calculate the number of iterations
        num_iterations = total_examples // batch_size
        if total_examples % batch_size != 0:
            num_iterations += 1  # Add one more iteration for the last, potentially smaller batch

        print(f"Number of iterations per epoch: {num_iterations}")'''
    def print_number_epochs(batch_size, tokenized_dataset):
        # Calculate the number of iterations needed for each epoch
        total_examples = len(tokenized_dataset)  # Total number of examples in the dataset

        # Calculate the number of iterations
        num_iterations = total_examples // batch_size
        if total_examples % batch_size != 0:
            num_iterations += 1  # Add one more iteration for the last, potentially smaller batch

        print(f"Number of iterations per epoch: {num_iterations}")

class GenerateStory():
    def generate_story(model, tokenized_prompt, max_length, device, start_symbol, k=10):
        model.eval()  # Set the model to evaluation mode

        src = torch.tensor([tokenized_prompt]).to(device)  # Convert to tensor and add batch dimension
        src_mask = (src != 0).unsqueeze(-2).to(device)  # Assuming 0 is the padding token
        output = TrainModel.greedy_decode(model, src, src_mask, max_len=max_length, start_symbol=start_symbol, k=k)

        return output

# Load JSON data and preprocess it
def load_and_preprocess_data(json_file, tokenizer):
    with open(json_file, 'r') as file:
        data = json.load(file)['train']['data']

    tokenized_data = []
    for item in data:
        encoded_text = tokenizer.encode(item['text'])
        tokenized_data.append({'encoded_text': encoded_text, 'tags': item['tags']})

    return tokenized_data



def load_and_preprocess_data(directory, tokenizer):
    tokenized_data = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file).get('train', {}).get('data', [])
                
                for item in data:
                    #encoded_text = tokenizer.encode(item['text'])
                    encoded_text = tokenizer.encode(item['text'], max_length=512, truncation=True)
                    tokenized_data.append({'encoded_text': encoded_text, 'tags': item['tags']})

    return tokenized_data




'''def load_and_preprocess_data(json_file, tokenizer):
    with open(json_file, 'r') as file:
        data = json.load(file)['train']['data']

    tokenized_data = []
    for item in data:
        encoded_text = tokenizer.encode(item['text'])
        tokenized_data.append({'encoded_text': encoded_text, 'tags': item['tags']})

    return tokenized_data
'''

def load_pretrained_tokenizer():
    # Initialize a pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return tokenizer

def save_checkpoint(model, optimizer, epoch, fold, iteration, max_checkpoints=5):
    # Use modulo operation to cycle through checkpoint indices (0 to max_checkpoints - 1)
    checkpoint_index = iteration % max_checkpoints
    filename = f'checkpoint_fold{fold+1}_index{checkpoint_index}.pth'

    checkpoint = {
        'epoch': epoch,
        'iteration': iteration, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

'''def load_and_preprocess_data(directory, tokenizer):
    tokenized_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file).get('train', {}).get('data', [])
                for item in data:
                    encoded_text = tokenizer.encode(item['text'], add_special_tokens=True)
                    tokenized_data.append({'encoded_text': encoded_text, 'tags': item['tags']})
    return tokenized_data
'''
'''
def train(model, train_data, criterion, optimizer, device, batch_size, fold, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        iteration = 0
        for batch in Helper.data_generator(train_data, batch_size, device):
            # Perform your training steps here
            # ...

            # Save a checkpoint every N iterations
            if iteration % 500 == 0:  # Adjust this value as needed
                save_checkpoint(model, optimizer, epoch, fold, iteration)
            iteration += 1

        print(f"Epoch {epoch+1} completed for fold {fold + 1}")
'''

def evaluate_model(model, test_data, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # No gradient updates
        for batch in Helper.data_generator(test_data, batch_size, device):
            # Extract the source, target, and masks from the batch
            src = batch.src
            trg = batch.trg
            src_mask = batch.src_mask
            trg_mask = batch.trg_mask
            trg_y = batch.trg_y
            ntokens = batch.ntokens

            # Forward pass
            outputs = model(src, trg, src_mask, trg_mask)

            # Calculate loss - remove norm (ntokens) from the arguments
            loss = loss_compute(outputs, trg_y, ntokens, is_train=False)
            total_loss += loss.item()
            total_tokens += ntokens.item()

    avg_loss = total_loss / total_tokens
    return avg_loss




def predict(model, X_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        predictions = model(X_test)  # Get the model's predictions
        # Convert predictions to the desired format, e.g., applying a threshold for classification
        predicted_labels = torch.argmax(predictions, dim=1)
    return predicted_labels

def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy



def monitor_training(epoch, train_loss, val_loss, overfit_threshold=0.05, underfit_threshold=0.1):
    """
    Monitor for signs of overfitting and underfitting after each epoch.
    :param epoch: Current epoch number.
    :param train_loss: Average training loss for the epoch.
    :param val_loss: Average validation loss for the epoch.
    :param overfit_threshold: Threshold for detecting overfitting.
    :param underfit_threshold: Threshold for detecting underfitting.
    """
    print(f"Epoch {epoch}: Training Loss: {train_loss}, Validation Loss: {val_loss}")

    # Detect overfitting
    if train_loss < val_loss - overfit_threshold:
        print("Warning: Potential overfitting detected.")
        # Additional logic or suggestions can be added here

    # Detect underfitting
    elif train_loss > underfit_threshold and val_loss > underfit_threshold:
        print("Warning: Potential underfitting detected.")
        # Additional logic or suggestions can be added here

def remove_repetitive_sentences(text):
    """ Remove repetitive sentences from the text """
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences)

def refine_story(model, initial_prompt, iterations, max_length, device, start_symbol_id):
    """ Iteratively refine the story """
    current_prompt = initial_prompt
    for _ in range(iterations):
        # Generate story
        tokenized_prompt = tokenizer.encode(current_prompt)
        generated_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length, device, start_symbol_id)
        generated_story = tokenizer.decode(generated_tokens.tolist()[0])

        # Post-process the story
        processed_story = remove_repetitive_sentences(generated_story)

        # Use the processed story as the new prompt
        current_prompt = processed_story
    
    return processed_story

# Number of folds
k = 5

# Create a KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=42)

#tokenizer = CustomTokenizer("tiny_stories_tokenizer.json")
tokenizer = load_pretrained_tokenizer()
i = 0
maxLoopNumber = 10
trainModel = True
#tokenizer = CustomTokenizer("tiny_stories_tokenizer.json")
# Get vocabulary sizes
src_vocab = tokenizer.vocab_size
tgt_vocab = tokenizer.vocab_size
print("Vocab size: ", src_vocab)
num_epochs = 10  # Number of epochs
N = 6  # Number of layers 
d_model = 512  # Dimension of the model
d_ff = 2048  # Dimension of feed forward layer
h = 8  # Number of heads
dropout = 0.1  # Dropout rate
device = Helper.get_device()
start_symbol_token = '[CLS]'  # or '[CLS]' depending on your model's training
start_symbol_id = tokenizer.vocab[start_symbol_token]
createModel = False
# Initialize model to None
model = None

# Get the parent directory of the current working directory
parent_directory = os.path.dirname(os.getcwd())
# Construct the path
directory_path_for_training = os.path.join(parent_directory, "books", "datasets_training")
directory_path_for_testing = os.path.join(parent_directory, "books", "datasets_test")

tokenized_data = load_and_preprocess_data(directory_path_for_training, tokenizer)
tokenized_data_training = load_and_preprocess_data(directory_path_for_testing, tokenizer)

print(len(tokenized_data))
print(len(tokenized_data_training))


tokenized_data = np.array(tokenized_data)  # Convert to a NumPy array for easy indexing
test_data = np.array(tokenized_data_training) # Convert to a NumPy array for easy indexing
batch_size = 1 # Set a suitable batch size
createModel = False
train = False

if (train):
    for fold, (train_index, test_index) in enumerate(kf.split(tokenized_data)):
        print(f"Running fold {fold + 1}/{k}")
        
        # Split data into training and test set for this fold
        #train_data, test_data = tokenized_data[train_index], tokenized_data[test_index]

        model = MakeModel.make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)

        if (createModel == False):
            # Load the model
            model.load_state_dict(torch.load('model.pth'))
            #checkpoint = torch.load('model.pth')
            print("Model loaded from model.pth")
            #model.load_state_dict(checkpoint['model_state_dict'])
            #print(torch.cuda.is_available())
            #prompt = "Tim wanted to"  # Your starting text
            #print("Prompt:", prompt)
            #tokenized_prompt = tokenizer.encode(prompt)
            #generated_story_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length=100, device=device, start_symbol=start_symbol_id)
            #generated_story = tokenizer.decode(generated_story_tokens.tolist()[0])
            #print(generated_story)

        model = model.to(device)

        # Loss and Optimizer for this fold
        criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.1)
        optimizer = NoamOpt.get_std_opt(model)
        scheduler = ReduceLROnPlateau(optimizer.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

        # Helper function to print number of epochs
        Helper.print_number_epochs(batch_size, tokenized_data)

        # Training loop for this fold
        iteration = 0
        best_val_loss = float('inf')
        patience = 3
        trigger_times = 0
        for epoch in range(num_epochs):
            model.train()
            loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
            avg_train_loss = TrainModel.run_epoch(Helper.data_generator(tokenized_data, batch_size, device), model, loss_compute, optimizer, epoch, fold)
            model.eval()
            avg_val_loss = evaluate_model(model, test_data, criterion, device)  # You need to implement this function to calculate validation loss
            # Update the scheduler with the validation loss
            scheduler.step(avg_val_loss)
            # Monitor the training process for overfitting/underfitting
            monitor_training(epoch, avg_train_loss, avg_val_loss)
            print(f"Epoch {epoch + 1}: Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            # Save the final model
            torch.save(model.state_dict(), 'model.pth')
            print("Model saved as model.pth")

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            # Evaluate on test data
            #test_loss, test_accuracy = evaluate_model(model, test_data, criterion, device)
            #print(f"Epoch {epoch+1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            # Optionally, you can evaluate the model on test_data here
            #predictions = model.predict(X_test)
            # Calculate accuracy or other metrics
            #accuracy = accuracy_score(y_test, predictions)
            #print(f"Accuracy: {accuracy}")

            # For a more detailed report
            #print(classification_report(y_test, predictions))
else:
    model = MakeModel.make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()
    initial_prompt = "Narrate a day in the life of a young woman in the 1800s, detailing her aspirations, familial duties, and the societal expectations she navigates."  # Your starting text
    print("Prompt:", initial_prompt)
    tokenized_prompt = tokenizer.encode(initial_prompt)
    refined_story = refine_story(model, initial_prompt, 3, 500, device, start_symbol_id)
    #generated_story_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length=100, device=device, start_symbol=start_symbol_id)
    #generated_story = tokenizer.decode(generated_story_tokens.tolist()[0])
    print(refined_story)