import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

############# Setup and data loading ################

# Load the Swedish spaCy model
nlp = spacy.load('sv_core_news_lg')

data = pd.read_csv('../data/Friends_data_only_user_annotation.csv')

# Split into text and labels
texts = data['text'].values
labels = data['label text'].values


############## Preprocessing ################

# Read file with Swedish stopwords
# https://github.com/stopwords-iso/stopwords-sv
# https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages
def load_stopwords(file_path):
    words_lst = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for word in file.readlines():
            words_lst.append(word.strip())
    return set(words_lst)

# Load stop words
stop_words_file = '../data/swedish_stopwords.txt'
stop_words = load_stopwords(stop_words_file)


def preprocess_texts(texts, stop_words):
    processed_texts = []
    
    for text in texts:
        if not isinstance(text, str):
            if pd.isnull(text):
                text = ""
            else:
                text = str(text)
        
        doc = nlp(text)
        tokens = [token.text for token in doc if token.text.lower() not in stop_words and len(token.text) > 2]
        processed_texts.append(" ".join(tokens))


        


    
    return processed_texts

swedish_sentences = [
    "Hon går ofta till biblioteket för att låna böcker.",
    "Vädret var perfekt för en lång promenad idag.",
    "Katten sover i solen på fönsterbrädan.",
    "Vi måste handla mer kaffe, vi är nästan slut.",
    "Han! lär sig spela! gitarr genom onlinekurser mobbar!."
    ":) :)",
    "....",
    ".",
    ":)",
    "alla hotar mig",
    "ingen hotar mig"
]




l = preprocess_texts(swedish_sentences, stop_words)

for i in l:
    print(i)

