import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import encoder as Encoder
import clones as Clones
import layerNorm as LayerNorm
import sublayerConnection as SublayerConnection

class EncoderDecorders(nn.Module):
    # A standard Encoder-Decoder architecture. Base for this and many other models.
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(encoder.EncoderDecorder, self).__init__()
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
    
class EncoderLayer(nn.Module):
    # Encoder is made up of two layers: multi-head self-attn and position-wise feed forward network
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Clones.clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # Follow Figure 1 (left) for connections.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
