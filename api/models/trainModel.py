import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time 
from torch.autograd import Variable
import torch
from datasets import load_dataset
from services import Helper

class TrainModel():
    def run_epoch(data_iter, model, loss_compute):  
        # Standard Training and Logging Function
        # Generate a training and scoring function to keep track of loss.
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            Helper.print_memory_usage()
            print(i)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i%50==1:
                elapsed = time.time()-start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss/batch.ntokens, tokens/elapsed))
                start = time.time()
                tokens = 0
        return total_loss/total_tokens

    def batch_size_fn(new, count, sofar):
        # Keep augmenting batch and calculate total number of tokens + padding.
        global max_src_in_batch, max_tgt_in_batch
        if count==1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, len(new.src))
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg)+2)
        src_elements = count*max_src_in_batch
        tgt_elements = count*max_tgt_in_batch
        return max(src_elements, tgt_elements)
    
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        # Greedy decode function.
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            #out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            out = model.decode(memory, src_mask, Variable(ys), Variable(DecoderLayer.subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys