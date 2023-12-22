import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time 
from torch.autograd import Variable
import torch
from datasets import load_dataset
import attention as attention
import positionwiseFeedForward as PositionwiseFeedForward
import positionalEncoding as PositionalEncoding
import encoder as Encoder

class MakeModel():
    # Helper: Construct a model from hyperparameters.
    def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        attn = attention.MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = Encoder.EncoderDecorder(Encoder(Encoder.EncoderLayer(d_model, c(attn), c(ff), dropout), N), Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), nn.Sequential(Embeddings(d_model, src_vocab), c(position)), nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), Generator(d_model, tgt_vocab))
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        return model