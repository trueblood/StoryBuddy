import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    # Use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.
    def __init__(self, d_model, vocab):
        # Embedding layer.
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)