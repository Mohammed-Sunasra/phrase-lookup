# phrase-lookup project
A medical dictionary based lookup project based on keyword/phrase matching

## Steps in running the model:
1. Create a virtual environment  
2. Install dependencies mentioned in the requirements.txt file
3. Create a folder named "model_files" inside src/ directory that will store all the model related files.
4. Create a folder named "ouput_files" inside src/ directory that will store all data related files.
5. run "wget http://nlp.stanford.edu/data/glove.6B.zip" to download word embeddings.
6. Create a folder named word_embebddings and unzip the "glove.6B.zip" inside this folder

## Refactoring/Coding best practices: 
1. Separate out dataloading, preprocessing and modeling into different class based Modules.
2. Save data, tokenizer and model files into separate folders.
3. Write functions for every action that needs to be taken for eg preprocessing it, tokenizing it and finally passing the data to the model.
4. Build a pipeline of steps each doing 1 thing and passing it over to the next step in the model building process.
5. Experiment building models in Jupyter notebooks and once done, make the models into Modules that is reusable and reproducible.
