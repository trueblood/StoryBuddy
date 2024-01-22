import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from datasets import load_dataset

class Helper():
    def get_device():
        # Check whether GPU is available and use it if yes.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print(f"Using device: {device}")
        return device
    '''
    #This function is crucial for converting your tokenized dataset into a format that can be processed by the TrainModel.run_epoch function.
    def data_generator(tokenized_dataset, batch_size, device):
        # Function to yield batches of data
        for i in range(0, len(tokenized_dataset), batch_size):
            # Extract a batch of tokenized data
            batch_data = tokenized_dataset[i:i + batch_size]

            # Debugging print statement
            print("Batch data example:", batch_data[0])

            # Extracting src and assuming trg is the same as src for this example
            src_batch = [torch.tensor(item['src']) for item in batch_data]
            trg_batch = [torch.tensor(item['src']) for item in batch_data]  # Assuming target is same as source

            # Padding the sequences and converting to tensors
            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
            trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

            yield Batch(src_batch, trg_batch, 0)  # 0 is the padding index
            '''

    @staticmethod
    def data_generator(tokenized_dataset, batch_size, device):
        # Assuming tokenized_dataset is an iterable of dicts with 'src' key
        batch = []
        for item in tokenized_dataset:
            batch.append(item)
            if len(batch) == batch_size:
                src_batch = [torch.tensor(d['src']) for d in batch]
                trg_batch = [torch.tensor(d['src']) for d in batch]  # Replace with trg if available

                src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
                trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

                yield Batch(src_batch, trg_batch, 0)  # 0 is the padding index
                batch = []

        # Handle any remaining items in the batch
        if batch:
            src_batch = [torch.tensor(d['src']) for d in batch]
            trg_batch = [torch.tensor(d['src']) for d in batch]  # Replace with trg if available

            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
            trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0).to(device)

            yield Batch(src_batch, trg_batch, 0)  # 0 is the padding index

    def print_memory_usage():
        print(f"Current memory usage: {psutil.virtual_memory().percent}%")

    def print_number_epochs(batchSize):
        # this lets me know how many loops that will run
        total_examples = len(tokenized_dataset['train'])  # Total number of examples in the dataset
        batch_size = batchSize 

        # Calculate the number of iterations
        num_iterations = total_examples // batch_size
        if total_examples % batch_size != 0:
            num_iterations += 1  # Add one more iteration for the last, potentially smaller batch

        print(f"Number of iterations per epoch: {num_iterations}")