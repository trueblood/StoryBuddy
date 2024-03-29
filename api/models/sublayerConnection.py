import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from datasets import load_dataset
from .layerNorm import LayerNorm

class SublayerConnection(nn.Module):
    # A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last.
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Apply residual connection to any sublayer with the same size.
        return x+self.dropout(sublayer(self.norm(x)))
    