import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import clones as Clones
import layerNorm as LayerNorm
import sublayerConnection as SublayerConnection

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
