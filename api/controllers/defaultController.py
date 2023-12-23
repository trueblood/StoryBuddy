import torch
from services import GenerateStory
from models import CustomTokenizer
from models import makeModel as MakeModel
from services import Helper

class StoryController:
    def __init__(self):
        self.story_generator = GenerateStory()

    def generate_story(self, prompt, max_length):
        device = Helper.get_device()
        start_symbol_token = '<start>'
        start_symbol_id = tokenizer.vocab[start_symbol_token]
        # Load the model
        model = model.load_state_dict(torch.load('model.pth'))
        print("Model loaded from model.pth")
        tokenizer = CustomTokenizer("tiny_stories_tokenizer.json")
        tokenized_prompt = tokenizer.encode(prompt)
        generated_story_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length=max_length, device=device, start_symbol=start_symbol_id)
        generated_story = tokenizer.decode(generated_story_tokens.tolist()[0])
        return generated_story