import requests
import json
from dotenv import load_dotenv
import os

class Story:
    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def hash_id(self):
        return self._hash_id

    @hash_id.setter
    def hash_id(self, value):
        self._hash_id = value

class Twist:
    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def hash_id(self):
        return self._hash_id

    @hash_id.setter
    def hash_id(self, value):
        self._hash_id = value

    @property
    def parent_hash_id(self):
        return self._parent_hash_id

    @parent_hash_id.setter
    def parent_hash_id(self, value):
        self._parent_hash_id = value

    @property
    def twist_layer(self):
        return self._twist_layer

    @twist_layer.setter
    def twist_layer(self, value):
        self._twist_layer = value

def post_story(api_url, api_key, story):
    headers = {
        'x-auth-token': api_key,  # Changed from 'Authorization'
        'Content-Type': 'application/json'
    }

    data = {
        'title': story.title,
        'body': story.text
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 201 and response.content:
        try:
            return response.json()  # Expecting a JSON response with hashId
        except json.JSONDecodeError:
            return 'Invalid JSON response'
    else:
        return f'Error: {response.status_code} - {response.text}'
    
def post_twist(api_url, api_key, twist):
    headers = {
        'x-auth-token': api_key,  # Changed from 'Authorization'
        'Content-Type': 'application/json'
    }

    data = {
        'hashParentId': twist.parent_hash_id,
        'isExtraTwist': True,
        'title': twist.title,
        'body': twist.text
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 201 and response.content:
        try:
            return response.json()  # Expecting a JSON response with hashId
        except json.JSONDecodeError:
            return 'Invalid JSON response'
    else:
        return f'Error: {response.status_code} - {response.text}'

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    api_url_create_story = 'https://story3.com/api/v2/stories'
    api_url_create_twist = 'https://story3.com/api/v2/twists'

    api_key = os.getenv("API_KEY")

    # Get story details and post the story
    story_title = input("Enter your story title: ")
    story_body = input("Enter your story body: ")

    story = Story()
    story.title = story_title
    story.text = story_body

    result = post_story(api_url_create_story, api_key, story)
    print('Story Created')
    story.hash_id = result['hashId']
    
    twists = []  # Create an empty list to store the twists

    # Ask for the number of layers
    num_layers = int(input("Enter the number of layers you want to add: "))

    # Iterate through each layer
    for layer in range(1, num_layers + 1):
        print(f"--- Adding twists for Layer {layer} ---")

        if layer == 1:
            # For the first layer, add twists directly to the story
            twist_number = int(input(f"Enter the number of twists for Layer {layer}: "))
            for i in range(twist_number):
                twist_title = input(f"Enter Twist {i+1} title for Layer {layer}: ")
                twist_body = input(f"Enter Twist {i+1} body for Layer {layer}: ")
                twist = Twist()
                twist.title = twist_title
                twist.text = twist_body
                twist.twist_layer = layer
                twist.parent_hash_id = story.hash_id

                # Post the twist and append to the list
                result = post_twist(api_url_create_twist, api_key, twist)
                print(f'Twist {i+1} for Layer {layer} Created')
                twist.hash_id = result['hashId']
                twists.append(twist)
        else:
            # For layers beyond the first, add child twists for each parent twist
            parent_twists = [t for t in twists if t.twist_layer == layer - 1]
            for parent_twist in parent_twists:
                print(f"Adding child twists for parent twist: {parent_twist.title}")
                child_twist_number = int(input(f"Enter the number of child twists for '{parent_twist.title}': "))

                for i in range(child_twist_number):
                    twist_title = input(f"Enter child Twist {i+1} title for '{parent_twist.title}': ")
                    twist_body = input(f"Enter child Twist {i+1} body for '{parent_twist.title}': ")
                    twist = Twist()
                    twist.title = twist_title
                    twist.text = twist_body
                    twist.twist_layer = layer
                    twist.parent_hash_id = parent_twist.hash_id

                    # Post the twist and append to the list
                    result = post_twist(api_url_create_twist, api_key, twist)
                    print(f'Child Twist {i+1} for parent {parent_twist.title} Created')
                    twist.hash_id = result['hashId']
                    twists.append(twist)