import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time 
from torch.autograd import Variable
import torch
from datasets import load_dataset
from api.models.attention import Attention, MultiHeadedAttention
from api.models.positionwiseFeedForward import PositionwiseFeedForward
from api.models.positionalEncoding import PositionalEncoding
from api.models.encoder import Encoder, EncoderDecorders, EncoderLayer
from api.models.decoder import Decoder, DecoderLayer
from api.models.embeddings import Embeddings
from api.models.generator import Generator

class MakeModel():
    # Helper: Construct a model from hyperparameters.
    def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        #attn = MultiHeadedAttention(h, d_model)  # Replace with your actual attention class
        #ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # Replace with your actual feedforward class
        #position = PositionalEncoding(d_model, dropout)  # Replace with your actual positional encoding class
        #model = EncoderDecorders(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), nn.Sequential(Embeddings(d_model, src_vocab), c(position)), nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), Generator(d_model, tgt_vocab))
        # Assuming c is a function for deep copying
        # Instantiate each component separately
        attn = MultiHeadedAttention(h, d_model)  # Replace with your actual attention class
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # Replace with your actual feedforward class
        position = PositionalEncoding(d_model, dropout)  # Replace with your actual positional encoding class
        try:        
            # Encoder and Decoder layers
            encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
            decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

            # Encoder and Decoder
            encoder = Encoder(encoder_layer, N)
            decoder = Decoder(decoder_layer, N)

            # Embedding layers for source and target
            src_embedding = nn.Sequential(Embeddings(d_model, src_vocab), c(position))  
            tgt_embedding = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
            # Generator
            generator = Generator(d_model, tgt_vocab)
        
            # Finally, instantiate the EncoderDecoder model
            model = EncoderDecorders(encoder, decoder, src_embedding, tgt_embedding, generator)
        except Exception as e:
            print(f"An error occurred: {e}")
        
        
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        return model