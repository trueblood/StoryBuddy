import torch.nn as nn
import copy 

class Clones(nn.Module):
    def clones(module, N):
        print('in clones')
        # Produce N identical layers.
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#class Clones(nn.Module):
#    def __init__(self, module, N):
#        super(Clones, self).__init__()
        # Produce N identical layers.
 #       self.clones = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
