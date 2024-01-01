import spacy
import json
import re
from pathlib import Path
import os
import re
from sklearn.model_selection import KFold
import glob



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

'''
def create_kfold_datasets(file_path, keywords, k=5):
    # Load the book text
    book_text = load_text(file_path)
    metadata, chapters = extract_metadata_and_chapters(book_text)  # Assuming this function extracts metadata and chapters

    kf = KFold(n_splits=k)
    fold_datasets = []

    for train_index, test_index in kf.split(chapters):
        train_chapters = [chapters[i] for i in train_index]
        test_chapters = [chapters[i] for i in test_index]

        train_dataset = create_dataset(train_chapters, metadata, keywords)
        test_dataset = create_dataset(test_chapters, metadata, keywords)

        fold_dataset = {
            'train': train_dataset,
            'test': test_dataset
        }
        fold_datasets.append(fold_dataset)

    return fold_datasets  # Returns a list of dictionaries for each fold
'''
'''
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
        'metadata': metadata,
        'data': {
            'training': chapter_data,
            'validation': [],
            'test': []
        }
    }

    return json.dumps(dataset, indent=4, ensure_ascii=False)
'''
def create_kfold_datasets(keywords, k, book):
    # Load the book text
    book_text = load_text(book.path)

    # Extract metadata
    metadata = create_metadata(book_text)

    # Extract chapters
    chapters_text = extract_content(book_text, 'c')  # Assuming this function works as intended
    chapters = [{'chapter': idx + 1, 'text': chapter.strip()} for idx, chapter in enumerate(chapters_text)]

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True)
    fold_datasets = []

    #for train_index, test_index in kf.split(chapters):
    for fold_index, (train_index, test_index) in enumerate(kf.split(chapters)):
        print(f"Fold {fold_index + 1}:")
        print("Training chapter indices:", train_index)
        print("Testing chapter indices:", test_index)

        # Splitting chapters into training and testing
        train_chapters = [chapters[i] for i in train_index]
        test_chapters = [chapters[i] for i in test_index]

        # Optionally, print some content of train and test chapters to inspect
        print("Sample from first training chapter:", train_chapters[0]['text'][:100])  # Print first 100 chars
        print("Sample from first testing chapter:", test_chapters[0]['text'][:100])  # Print first 100 chars


        # Create datasets for each fold
        #train_dataset = process_chapters(train_chapters, keywords)
        #test_dataset = process_chapters(test_chapters, keywords)
        train_dataset = {'metadata': metadata, 'data': process_chapters(train_chapters, keywords)}
        test_dataset = {'metadata': metadata, 'data': process_chapters(test_chapters, keywords)}

        fold_dataset = {
            'train': train_dataset,
            'test': test_dataset
        }
        fold_datasets.append(fold_dataset)

    return fold_datasets

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

def process_chapters(chapters, keywords):
    processed_data = []
    for chapter in chapters:
        chapter_text = clean_text(chapter['text'])
        sections = split_text_into_sections(chapter_text)

        for section in sections:
            # Use the identify_custom_tags function to tag the section
            tags = identify_custom_tags(section, keywords)

            processed_data.append({
                'chapter': chapter['chapter'],
                'section': sections.index(section) + 1,
                'text': section,
                'tags': tags
            })
    return processed_data

class Book:
    def __init__(self, path):
        self.path = path
        self.filename_with_extension = os.path.basename(path)
        self.filename = os.path.splitext(os.path.basename(path))[0]

# Specify the path to your .txt file
file = Path('../books/pg28587.txt')
file_path = os.path.join(os.path.dirname(file), "texts")
txt_files = glob.glob(os.path.join(file_path, "*.txt"))
books = [Book(file) for file in txt_files]

if len(books) > 0:
    for book in books:
        
        k = 5
        # Check if the file exists and create the dataset
        #if not txtFile.is_file():
        #    print(f"The file {txtFile} does not exist.")
        #else:
        #json_dataset = create_dataset(file_path, keywords)
        # Get the path to the "datasets" folder

        #texts_folder = os.path.join(os.path.dirname(file_path), "texts")
        #txt_files = glob.glob(os.path.join(texts_folder, "*.txt"))

        datasets_training_folder = os.path.join(os.path.dirname(file_path), "datasets_training")
        datasets_test_folder = os.path.join(os.path.dirname(file_path), "datasets_test")

        # Create the "datasets" folder if it doesn't exist
        os.makedirs(datasets_training_folder, exist_ok=True)
        os.makedirs(datasets_test_folder, exist_ok=True)

        # Iterate over each text file
        #for txt_file in txt_files:
        #base_filename = os.path.splitext(os.path.basename(txtFile))
        # Set the output file path within the "datasets" folder
        #output_file_path = os.path.join(datasets_folder, file_path.stem + ".json")
        kfold_datasets = create_kfold_datasets(keywords, k, book)

        # Save the dataset to a JSON file
        #with open(output_file_path, 'w', encoding='utf-8') as json_file:
            #json_file.write(json_dataset)

        # Create the output directory if it doesn't exist
        #os.makedirs(output_file_path, exist_ok=True)
        
        # Iterate over each fold in the kfold_datasets
        for fold_index, fold_data in enumerate(kfold_datasets):
            print('in kfold loop')
            # Construct the filename for each fold
            base_filename = 'dataset'  # Base name for your files
            filename = f"{base_filename}_{book.filename}_fold{fold_index+1}.json"
            if fold_index < (k - 1):
                file_path = os.path.join(datasets_training_folder, filename)
            else:
                file_path = os.path.join(datasets_test_folder, filename)

            # Write the fold data to a JSON file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(fold_data, file, indent=4, ensure_ascii=False)

            print(f"Saved Fold {filename} dataset to {file_path}")

            