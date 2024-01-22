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