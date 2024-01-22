import numpy as np
import torch
import torch.nn as nn
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
from models import customTokenizer as CustomTokenizer
import helper as Helper
from model import makeModel as MakeModel
from model import labelSmoothing as LabelSmoothing
from model import noamOpt as NoamOpt
from model import simpleLossCompute as SimpleLossCompute
from model import trainModel as TrainModel
from model import generateStory as GenerateStory
   
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

# test this code
tokenizer = CustomTokenizer("tiny_stories_tokenizer.json")
# Get vocabulary sizes
src_vocab = tokenizer.get_vocab_size()
tgt_vocab = tokenizer.get_vocab_size()
org_dataset = load_dataset("roneneldan/TinyStories")
# shuffle
dataset = org_dataset.shuffle(seed=42)
#print("Columns in the dataset:", dataset['train'].column_names)
num_epochs = 10  # Number of epochs
N = 6  # Number of layers
d_model = 512  # Dimension of the model
d_ff = 2048  # Dimension of feed forward layer
h = 8  # Number of heads
dropout = 0.1  # Dropout rate
# Select a smaller subset of the dataset
num_examples = min(500, len(dataset['train']))
dataset['train'] = dataset['train'].select(range(num_examples))
tokenized_dataset = dataset.map(CustomTokenizer.tokenize_fn, batched=True)
device = Helper.get_device()
batch_size = 1 # Set a suitable batch size
model = MakeModel.make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)
model = model.to(device) #move model to appropriate device
# Loss and Optimizer
criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.1)
optimizer = NoamOpt.get_std_opt(model)
print("Dataset example:", tokenized_dataset['train'][0])
start_symbol_token = '<start>'  # or '[CLS]' depending on your model's training
start_symbol_id = tokenizer.vocab[start_symbol_token]
print("Start symbol id:", start_symbol_id)

trainModel = False

if (trainModel): # seems to run 500 times before next epoch
    # Load the model
    model.load_state_dict(torch.load('model.pth'))
    print("Model loaded from model.pth")
    Helper.print_number_epochs(batch_size)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
        TrainModel.run_epoch(Helper.data_generator(tokenized_dataset['train'], batch_size, device), model, loss_compute)
        model.eval()
        # Evaluate the model on validation data if available
        # Save the final model
        torch.save(model.state_dict(), 'model.pth')
        print("Model saved as model.pth")
else:
    # Assuming model is an instance of the correct class
    model = MakeModel.make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)
    model = model.to(device) #move model to appropriate device

    # Load the model
    model.load_state_dict(torch.load('model.pth'))
    print("Model loaded from model.pth")

print(torch.cuda.is_available())
prompt = "Lilly wanted to go to the mall"  # Your starting text
print("Prompt:", prompt)
tokenized_prompt = tokenizer.encode(prompt)
generated_story_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length=300, device=device, start_symbol=start_symbol_id)
generated_story = tokenizer.decode(generated_story_tokens.tolist()[0])
print(generated_story)

