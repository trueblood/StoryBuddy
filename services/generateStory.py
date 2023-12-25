import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from datasets import load_dataset
from .trainModel import TrainModel

class GenerateStory():
    def generate_story(model, tokenized_prompt, max_length, device, start_symbol):
        model.eval()  # Set the model to evaluation mode
        src = torch.tensor([tokenized_prompt]).to(device)  # Convert to tensor and add batch dimension

        src_mask = (src != 0).unsqueeze(-2).to(device)  # Assuming 0 is the padding token
        print("in generate story")
        output = TrainModel.greedy_decode(model, src, src_mask, max_len=max_length, start_symbol=start_symbol)

        return output