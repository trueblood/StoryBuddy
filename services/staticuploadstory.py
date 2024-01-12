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

def get_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_story(file):
    story = Story()
    story.title = file.get('title')
    print(story.title)
    story.text = file.get('text')
    print(story.text)
    return story

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

def publish_twist(api_url_base, api_key, hash_id):
    api_url = f"{api_url_base}/{hash_id}/publish"
    headers = {
        'x-auth-token': api_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers)

    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            return 'Invalid JSON response'
    else:
        return f'Error: {response.status_code} - {response.text}'


def extract_twists_and_child_twists(data, result=None):
    """
    Recursively extracts all child twists from the given data.
    :param data: The JSON data containing the twists.
    :param result: A list to accumulate the results.
    :return: A list of all child twists.
    """
    if result is None:
        result = []

    if isinstance(data, list):
        for item in data:
            result.append(item)  # Append the twist itself
            if 'childTwists' in item:
                extract_twists_and_child_twists(item['childTwists'], result)

    return result


if __name__ == "__main__":
    load_dotenv()
    api_url_create_story = 'https://story3.com/api/v2/stories'
    api_url_create_twist = 'https://story3.com/api/v2/twists'
    api_url_base_twists = 'https://story3.com/api/v2/twists'
    api_key = os.getenv("API_KEY")

    json_file_path = input("Enter the path to the JSON file: ")
    file = get_data(json_file_path)

    story = get_story(file.get('story'))
    story_result = post_story(api_url_create_story, api_key, story)
    print(story_result)
    story.hash_id = story_result.get('hashId')
    print('Story Created')

    twists = extract_twists_and_child_twists(file['twists'])
    print(twists[:5])
