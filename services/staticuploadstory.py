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
    def child_twists(self):
        return self._child_twists

    @child_twists.setter
    def child_twists(self, value):
        self._child_twists = value

class ChildTwist(Twist):
    def __init__(self, title, text):
        super().__init__(title, text)

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


def extract_twists_and_child_twists_post_publish(data, story, result=None):
    api_url_create_story = 'https://story3.com/api/v2/stories'
    api_url_create_twist = 'https://story3.com/api/v2/twists'
    api_url_base_twists = 'https://story3.com/api/v2/twists'
    api_key = os.getenv("API_KEY")
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
            twist = Twist()
            twist.title = item.get('title')
            twist.text = item.get('text')
            # Post the twist and append to the list
            twist.parent_hash_id = story.hash_id
            result = post_twist(api_url_create_twist, api_key, twist)
            twist.hash_id = result['hashId']
            publish_result = publish_twist(api_url_base_twists, api_key, twist.hash_id)
            print(twist.title)
            if 'childTwists' in item:
                for child_item in item['childTwists']:
                    child_twist = Twist()
                    child_twist.parent_hash_id = twist.hash_id
                    child_twist.title = child_item.get('title')
                    child_twist.text = child_item.get('text')
                    # Post the twist and append to the list
                    result = post_twist(api_url_create_twist, api_key, child_twist)
                    child_twist.hash_id = result['hashId']
                    publish_result = publish_twist(api_url_base_twists, api_key, child_twist.hash_id)
                    print(child_twist.title)
                    for grandchild_item in child_item['childTwists']:
                        grandchild_twist = Twist()
                        grandchild_twist.parent_hash_id = child_twist.hash_id
                        grandchild_twist.title = grandchild_item.get('title')
                        grandchild_twist.text = grandchild_item.get('text')
                        # Post the twist and append to the list
                        result = post_twist(api_url_create_twist, api_key, grandchild_twist)
                        grandchild_twist.hash_id = result['hashId']
                        publish_result = publish_twist(api_url_base_twists, api_key, grandchild_twist.hash_id)
                        print(grandchild_twist.title)
                        for greatgrandchild_item in grandchild_item['childTwists']:
                            greatgrandchild_twist = Twist()
                            greatgrandchild_twist.parent_hash_id = grandchild_twist.hash_id
                            greatgrandchild_twist.title = greatgrandchild_item.get('title')
                            greatgrandchild_twist.text = greatgrandchild_item.get('text')
                            # Post the twist and append to the list
                            result = post_twist(api_url_create_twist, api_key, greatgrandchild_twist)
                            greatgrandchild_twist.hash_id = result['hashId']
                            publish_result = publish_twist(api_url_base_twists, api_key, greatgrandchild_twist.hash_id)
                            print(greatgrandchild_twist.title)
                            #twist.child_twists.append(greatgrandchild_twist)
                        #twist.child_twists.append(grandchild_twist)
                    #child_twist = ChildTwist(child_item.get('title'), child_item.get('text'))
                    #print(child_twist.title)
                    #print(child_twist.text)
                    #twist.child_twists.append(child_twist)
                    # Recursively extract child twists of this child twist
                    #extract_twists_and_child_twists(child_item.get('childTwists', []), result)

            #result.append(item)  # Append the twist itself
            #if 'childTwists' in item:
            #    twist.child_twists.append(item['childTwists'])
            #    extract_twists_and_child_twists(item['childTwists'], result)

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

    twists = extract_twists_and_child_twists_post_publish(file['twists'], story)



    print(twists[:5])
