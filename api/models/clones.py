import torch.nn as nn
import copy 

class Clones(nn.Module):
    def clones(module, N):
        # Produce N identical layers.
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])