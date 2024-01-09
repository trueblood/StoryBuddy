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

    story_title = input("Enter your story title: ")
    story_body = input("Enter your story body: ")

    story = Story()
    story.title = story_title
    story.text = story_body

    result = post_story(api_url_create_story, api_key, story)
    print('Story Created')
    story.hash_id = result['hashId']
    
    twists = []  # Create an empty list to store the twists

    twist_number = input("Enter the number of twists you want to add: ")
    twist_number = int(twist_number)
    for i in range(twist_number):
        twist_title = input(f"Enter your twist {i} title: ")
        twist_body = input(f"Enter your twist {i} body: ")
        twist = Twist()
        twist.title = twist_title
        twist.text = twist_body
        twist.parent_hash_id = story.hash_id
        twist.twist_layer = 1
        result = post_twist(api_url_create_twist, api_key, twist)
        print(f'Twist {i} Created: ')
        twist.hash_id = result['hashId']
        twists.append(twist)  # Append the twist to the list

    filtered_twists = [twist for twist in twists if twist.twist_layer == 1]
    for h in filtered_twists:
        print(f"in twist layer {h.twist_layer} twist title: {h.title}")
        twist_number = input("Enter the number of twists you want to add: ")
        twist_number = int(twist_number)
        for i in range(twist_number):
            twist_title = input(f"Enter your twist {i} title: ")
            twist_body = input(f"Enter your twist {i} body: ")
            twist = Twist()
            twist.title = twist_title
            twist.text = twist_body
            twist.parent_hash_id = h.hash_id
            twist.twist_layer = 2
            result = post_twist(api_url_create_twist, api_key, twist)
            print(f'Twist {i} Created')
            twist.hash_id = result['hashId']
            twists.append(twist)

    filtered_twists = [twist for twist in twists if twist.twist_layer == 2]
    for h in filtered_twists:
        print(f"in twist layer {h.twist_layer} twist title: {h.title}")
        twist_number = input("Enter the number of twists you want to add: ")
        twist_number = int(twist_number)
        for i in range(twist_number):
            twist_title = input(f"Enter your twist {i} title: ")
            twist_body = input(f"Enter your twist {i} body: ")
            twist = Twist()
            twist.title = twist_title
            twist.text = twist_body
            twist.parent_hash_id = h.hash_id
            twist.twist_layer = 3
            result = post_twist(api_url_create_twist, api_key, twist)
            print(f'Twist {i} Created')
            twist.hash_id = result['hashId']
            twists.append(twist)
    
    filtered_twists = [twist for twist in twists if twist.twist_layer == 3]
    for h in filtered_twists:
        print(f"in twist layer {h.twist_layer} twist title: {h.title}")
        twist_number = input("Enter the number of twists you want to add: ")
        twist_number = int(twist_number)
        for i in range(twist_number):
            twist_title = input(f"Enter your twist {i} title: ")
            twist_body = input(f"Enter your twist {i} body: ")
            twist = Twist()
            twist.title = twist_title
            twist.text = twist_body
            twist.parent_hash_id = h.hash_id
            twist.twist_layer = 4
            result = post_twist(api_url_create_twist, api_key, twist)
            print(f'Twist {i} Created')
            twist.hash_id = result['hashId']
            twists.append(twist)