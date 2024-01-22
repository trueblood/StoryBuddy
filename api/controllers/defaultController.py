import torch
from services.generateStory import GenerateStory
from ..models.customTokenizer import CustomTokenizer
from services.helper import Helper
import requests
import os
from flask import current_app
from services.makeModel import MakeModel 

class defaultController():
    def __init__(self):
        self.story_generator = GenerateStory()

    def generate_story(self, prompt, max_length, recaptcha_response):
        try:
            #prompt = 'Once upon a time'
            #max_length = 100
        #recaptcha_response = self.verify_recaptcha(recaptcha_response)
        #if recaptcha_response:
            device = Helper.get_device()
            tokenizer_path = os.path.join(current_app.root_path, 'services', 'tiny_stories_tokenizer.json')
            tokenizer = CustomTokenizer(tokenizer_path)
            #tokenizer_path = os.path.abspath('/media/squeebit/ExternalSSD/Tokenizes/tiny_stories_tokenizer.json')
            #tokenizer = CustomTokenizer(tokenizer_path)
            start_symbol_token = '<start>'
            start_symbol_id = tokenizer.vocab[start_symbol_token]

            # Define hyperparameters for the model
            src_vocab = tokenizer.get_vocab_size()
            tgt_vocab = tokenizer.get_vocab_size()
            num_epochs = 10  # Number of epochs
            N = 6  # Number of layers 
            d_model = 512  # Dimension of the model
            d_ff = 2048  # Dimension of feed forward layer
            h = 8  # Number of heads
            dropout = 0.1  # Dropout rate
            device = Helper.get_device()

            # Load the model 
            model = MakeModel.make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)
            #model_path = os.path.abspath('/media/squeebit/ExternalSSD/Models/model.pth')
            model_path = os.path.join(current_app.root_path, 'services', 'model.pth')


            #model = torch.load(model_path)
            #model = torch.load_state_dict(torch.load('model.pth'))
            #model.load_state_dict(torch.load('model.pth'))
            # Get the path to the current file
            #current_file_path = os.path.dirname(os.path.realpath(__file__))

            # Construct the path to the model.pth file
            #model_path = os.path.join(current_file_path, 'model.pth')

            #state_dict = torch.load(model_path)
            #print(state_dict.keys())


            try:
                model.load_state_dict(torch.load(model_path))
                
                #model.load_state_dict(torch.load(model_path))
                #model.load_state_dict(torch.load('model.pth'))
                #print('keys')
                #for key in model.state_dict().keys():
                #    print(key)

            except Exception as e:
                print(f"An error occurred: {e}")
            model = model.to(device) #move model to appropriate device
            model.eval()
            print(f"Model loaded from {model_path}")
            
            # Process the prompt
            tokenized_prompt = tokenizer.encode(prompt)
            generated_story_tokens = GenerateStory.generate_story(model, tokenized_prompt, max_length=max_length, device=device, start_symbol=start_symbol_id)
            generated_story = tokenizer.decode(generated_story_tokens.tolist()[0])
            return generated_story
        #else:
         #  return "Failed captcha"
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"
    
    def verify_recaptcha(self, recaptcha_response):
        print(recaptcha_response)
        secret_key = ''  # Replace with your secret key
        data = {
            'secret': secret_key,
            'response': recaptcha_response
        }
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data)
        result = response.json()
        return result.get('success', False)

    # In your route handling the form submission
   # @app.route('/fetch_data', methods=['POST'])
   # def handle_form_submission():
   #     data = request.json
   #     recaptcha_response = data.get('recaptcha_response')
   #     if not verify_recaptcha(recaptcha_response):
   #         return jsonify({'error': 'Invalid reCAPTCHA. Please try again.'}), 400
