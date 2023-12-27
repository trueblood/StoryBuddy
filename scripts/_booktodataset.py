import spacy
import json
import re
from pathlib import Path

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

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
def identify_custom_tags(text, keywords):
    # Process the text with spaCy
    doc = nlp(text)
    tags = []
    for token in doc:
        for tag, keyword_list in keywords.items():
            if token.lemma_ in keyword_list:
                tags.append(tag)
    return list(set(tags))

# Function to load text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
# Function to extract content based on tags
def extract_content(text, tag):
    pattern = re.compile(r'<{}>(.*?)</{}>'.format(tag, tag), re.DOTALL)
    return pattern.findall(text)

# Function to create the dataset
def create_dataset(file_path, keywords):
    # Load the book text
    book_text = load_text(file_path)

    # Extract metadata using the custom tags
    metadata = {
        'author': extract_content(book_text, 'a')[0].strip() if extract_content(book_text, 'a') else 'Unknown',
        'title': extract_content(book_text, 't')[0].strip() if extract_content(book_text, 't') else 'Unknown',
        'genre': extract_content(book_text, 'g')[0].strip() if extract_content(book_text, 'g') else 'Unknown',
        'source': extract_content(book_text, 's')[0].strip() if extract_content(book_text, 's') else 'Unknown',
        'publication_year': extract_content(book_text, 'p')[0].strip() if extract_content(book_text, 'p') else 'Unknown'
    }
    
    # Extract chapters
    chapters_text = extract_content(book_text, 'c')
    chapters = [{'chapter': idx + 1, 'text': chapter.strip()} for idx, chapter in enumerate(chapters_text)]

    # Process each chapter and identify tags
    chapter_data = []
    for i, chapter_text in enumerate(chapters):
        # Assume each chapter's first line is its title
        chapter_lines = chapter_text.strip().split('\n')
        chapter_title = chapter_lines[0].strip()
        chapter_body = ' '.join(chapter_lines[1:])
        
        # Identify tags using the custom function
        chapter_tags = identify_custom_tags(chapter_body, keywords)
        
        chapter_data.append({
            'chapter': i + 1,  # Chapter numbering starts at 1
            'title': chapter_title,
            'text': chapter_body,
            'tags': chapter_tags
        })

    # Package the data into the desired JSON structure
    dataset = {
        'metadata': metadata,
        'data': {
            'training': chapter_data,  # Assuming all chapters are for training
            'validation': [],  # Populate if you have validation chapters
            'test': []  # Populate if you have test chapters
        }
    }

    return json.dumps(dataset, indent=4, ensure_ascii=False)

# Specify the path to your .txt file
file_path = Path('/mnt/data/your_book_file.txt')

# Check if the file exists and create the dataset
if not file_path.is_file():
    print(f"The file {file_path} does not exist.")
else:
    json_dataset = create_dataset(file_path, keywords)
    
    # Save the dataset to a JSON file
    output_file_path = file_path.with_suffix('.json')
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_dataset)

    print(f"JSON dataset created successfully at {output_file_path}")
