from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_and_save_tokenizer(dataset_name, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    

    # Extract text data and prepare for tokenizer training
    texts = [story['text'] for story in dataset['train']]
    with open("tiny_stories_corpus.txt", "w", encoding="utf-8") as f:
        for story in texts:
            f.write(story + "\n")

    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer for the BPE model
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<start>", "<end>", "<eos>", " "])

    # Train the tokenizer
    tokenizer.train(["tiny_stories_corpus.txt"], trainer)

    # Save the tokenizer
    tokenizer.save(output_file)

    print(f"Tokenizer saved to {output_file}")

if __name__ == "__main__":
    # Define dataset name and output file
    DATASET_NAME = "roneneldan/TinyStories"
    OUTPUT_FILE = "tiny_stories_tokenizer.json"

    # Train and save the tokenizer
    train_and_save_tokenizer(DATASET_NAME, OUTPUT_FILE)
