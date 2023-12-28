import spacy
import json
import re
from pathlib import Path
import os
import re
from sklearn.model_selection import KFold

# Load the spaCy model for English
#nlp = spacy.load("en_core_web_sm")

# Define keywords for custom tags
keywords = {
    'Rome': ['Rome', 'Roman'],
    'girl': ['girl', 'daughter', 'maid', 'maiden'],
    'slavery': ['slavery', 'slave', 'enslaved'],
    'emperor': ['emperor', 'Caesar', 'ruler'],
    'family': ['family', 'kin', 'kinship', 'clan'],
    'friends': ['friend', 'companion', 'ally'],
    'sister': ['sister', 'sibling'],
    'Eleni': ['Eleni'],  # Assuming 'Eleni' is a proper noun and not a common word
    'fire': ['fire', 'flame', 'burn', 'blaze'],
    'female': ['female', 'woman', 'girl', 'lady'],
    'freedom': ['freedom', 'liberty', 'free']
}

# Function to identify custom tags based on keywords
# Uncomment and implement this function if you need to use it
'''
def identify_custom_tags(text, keywords):
    # Process the text with spaCy
    doc = nlp(text)
    tags = []
    for token in doc:
        for tag, keyword_list in keywords.items():
            if token.lemma_ in keyword_list:
                tags.append(tag)
    return list(set(tags))
'''

# Function to clean and split text into sections
def split_text_into_sections(text, min_length=1000, max_length=2000):
    sections = []
    while text:
        split_point = min(max_length, len(text) - 1)  # Ensure split_point is within the valid range
        # If the split_point is at the end of the text, just split here
        if split_point == len(text) - 1:
            sections.append(text.strip())
            break
        # Otherwise, find a suitable split point
        while split_point > min_length and text[split_point] not in [' ', '\n', '\t']:
            split_point -= 1

        sections.append(text[:split_point].strip())
        text = text[split_point:]

    return sections

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove newline characters and replace with a space
    text = text.replace('\n', ' ')
    # Normalize whitespace
    text = ' '.join(text.split())
    # Define the regex pattern for removing special characters
    # Keeping essential punctuation marks and numbers
    pattern = r'[^\w\s,.?!:;\'\"(){}\[\]\-]'
    # Remove special characters based on the defined pattern
    text = re.sub(pattern, '', text)
    return text

# Function to load text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to extract content based on tags
def extract_content(text, tag):
    pattern = re.compile(r'<{}>(.*?)</{}>'.format(tag, tag), re.DOTALL)
    return pattern.findall(text)

def create_kfold_datasets(file_path, k=5):
    # Load the book text
    book_text = load_text(file_path)
    metadata = create_metadata(book_text)
    chapters = create_chapters(book_text)

    kf = KFold(n_splits=k)
    fold_datasets = []
    for train_index, test_index in kf.split(chapters):
        train_chapters = [chapters[i] for i in train_index]
        test_chapters = [chapters[i] for i in test_index]

        train_dataset = create_dataset(train_chapters, metadata)
        test_dataset = create_dataset(test_chapters, metadata)

        fold_dataset = {
            'train': train_dataset,
            'test': test_dataset
        }
        fold_datasets.append(fold_dataset)
    return fold_datasets  # Returns a list of dictionaries for each fold

def create_metadata(book_text):
    # Extract metadata using the custom tags
    metadata = {
        'author': extract_content(book_text, 'a')[0].strip() if extract_content(book_text, 'a') else 'Unknown',
        'title': extract_content(book_text, 't')[0].strip() if extract_content(book_text, 't') else 'Unknown',
        'genre': extract_content(book_text, 'g')[0].strip() if extract_content(book_text, 'g') else 'Unknown',
        'source': extract_content(book_text, 's')[0].strip() if extract_content(book_text, 's') else 'Unknown',
        'publication_year': extract_content(book_text, 'p')[0].strip() if extract_content(book_text, 'p') else 'Unknown'
    }
    return metadata

def create_chapters(book_text):
    # Extract chapters
    chapters_text = extract_content(book_text, 'c')
    chapters = [{'chapter': idx + 1, 'text': chapter.strip()} for idx, chapter in enumerate(chapters_text)]
    return chapters

# Function to create the dataset
def create_dataset(metaData, chapters):
    # Process each chapter and identify tags
    chapter_data = []
    for i, chapter in enumerate(chapters):
        chapter_text = clean_text(chapter['text'])
        sections = split_text_into_sections(chapter_text)

        for section in sections:
            chapter_data.append({
                'chapter': i + 1,
                'section': sections.index(section) + 1,
                'text': section,
                'tags': 'null'  # Placeholder for tags
            })

   # Package the data into JSON structure
    dataset = {
        'metadata': metaData,
        'data': {
            'training': chapter_data,
            'validation': [],
            'test': []
        }
    }

    return json.dumps(dataset, indent=4, ensure_ascii=False)

# Specify the path to your .txt file
file_path = Path('../books/pg28587.txt')

# Check if the file exists and create the k-fold datasets
if not file_path.is_file():
    print(f"The file {file_path} does not exist.")
else:
    print(f"Creating datasets using the file at {file_path}")
    fold_datasets = create_kfold_datasets(file_path, k=5)
    #print(f"Created {len(fold_datasets)} datasets")
    datasets_folder = os.path.join(os.path.dirname(file_path), "datasets")
    os.makedirs(datasets_folder, exist_ok=True)

    for i, fold_dataset in enumerate(fold_datasets):
        #output_file_path = os.path.join(datasets_folder, f"{file_path.stem}_fold_{i}.json")
        output_file_path = os.path.join(datasets_folder, file_path.stem + ".json")
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(fold_dataset, indent=4, ensure_ascii=False))
            print(f"Dataset written to {output_file_path}")  # Confirm file creation
