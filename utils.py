import os

import pandas as pd 
import threading
import queue
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')
import nltk
nltk.download('wordnet')
nltk.download('words') 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))

def get_folders_in_folder(folder_path:str) -> list[str]:
    files_and_folders = os.listdir(folder_path)
    return [
        f
        for f in files_and_folders
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    
def get_files_inside_folder(folder_path:str)->list[str]:
    files_and_folders = os.listdir(folder_path)
    return [
        f
        for f in files_and_folders
        if os.path.isfile(os.path.join(folder_path, f))
    ]

def get_folder_array_data(folder_path:str,queue)->list[str]:
    files = get_files_inside_folder(folder_path)
    files_content = []
    for file in files:
        with open(f"{folder_path}/{file}") as file:
            files_content.append(file.read())
    queue.put((folder_path.split("/")[-1],files_content)) 
# took 1s
def get_dict_data_optimized(folder_path:str)->list[pd.DataFrame]:
    folders = get_folders_in_folder(folder_path)
    data =pd.DataFrame()
    threads = []
    my_queue = queue.Queue()
    for folder in folders :
        thread = threading.Thread(target=get_folder_array_data,args=(f"{folder_path}/{folder}",my_queue))
        thread.start()
        threads.append(thread)
    for thread in threads :
        thread.join()
        result = my_queue.get()
        df = pd.DataFrame(result[1],columns=["text"])
        df["label"] = result[0]
        
        data = pd.concat([data,df],axis = 0,ignore_index=True)
    return data

def short_words_eliminations(text:str):
    return re.sub(r'\b\w{1,2}\b', '', text)
def remove_punctuation_marks(text:str):
    return re.sub(r'[^\w\s]', '', text)
def remove_numbers(text:str):
    return re.sub(r'\d+', '', text)
def clean_text(row):
    
    # Remove stop words
    filtered_tokens = [token for token in row['tokens'] if token.lower() not in stop_words]
    clean_text = ' '.join(filtered_tokens)
    # Remove short workds
    clean_text = short_words_eliminations(clean_text)
    # Remove punctuation marks and special characters
    clean_text = remove_punctuation_marks(clean_text)
    # Remove numbers
    clean_text = remove_numbers(clean_text)
    # Remove not english words
    clean_text = remove_not_english_words(clean_text)
    return clean_text

def text_lemmatize(row):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(row["clean_text"])
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)



def remove_not_english_words(text:str):
    english_words = set(nltk.corpus.words.words())
    return ' '.join(word for word in text.split() if word.lower() in english_words)
    
