# StoryBuddy
An AI for Writing Stories

## Writing Story Outlines
Begin by crafting the structure of your stories:
- **Intro**: Set the stage for your narrative.
- **Body**: Develop the plot through several sections.
- **Conclusion**: Bring your story to a satisfying end.

## Extracting Keywords
- Pull out keywords from your story outlines.

## Sourcing Books
- Search public domain websites for books matching your keywords.

## Tagging Books
Use tags for data cleaning:
- `<d></d>`: Delete section.
- `<a></a>`: Author's name.
- `<t></t>`: Title of the work.
- `<g></g>`: Genre.
- `<s></s>`: Source of the book.
- `<p></p>`: Publication year.
- `<c></c>`: Content of the book.

## Preparing the Text Files
- Place the plain text book files in the `../books` directory.
- Ensure the files are in `.txt` format.

## Generating K-Fold Datasets
- Execute `_booktodataset_new_kfold.py` script located in the `scripts` folder to create the k-fold training datasets.

## Training the Model
- Once the dataset is prepared, start training the model.
- Run `_run_build_model.py` from the `scripts` folder. This script will reinitiate training if any interruptions occur.

## Post-Training Steps
- Move the trained model to the `services` folder.

## Launching the API
- Start the backend service by running `python3 main.py`.

## Starting the Web Application
- Navigate to the `StoryBuddyWeb` folder.
- Launch the application with `python3 app.py`.

## Publishing to story3.com
To add your story to story3.com:
- Prepare your story and twists in `story.json`, adhering to the provided structure.
- Rename the file to match the title of your story.
- Execute the `python3 staticuploadstory.py` script from the services directory.
- Follow the prompts to specify the path to your JSON file.
- Observe the console for errors during upload.
- Once uploaded, visit story3.com to officially publish your story.
- Share your story on various social media channels.