import stanza
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetimew

"""
This script include the preprocessing of the data using the stanza library.
It performs tokenization, lemmatization and stop words removal
The preprocessed data is then saved to a new CSV file.
"""


############# setup and data loading ################

# Ensure the Swedish model is downloaded
stanza.download('sv', verbose=False)

# Initialize the Stanza pipeline for Swedish, without the mwt processor
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma', verbose=False, use_gpu=False)

data = pd.read_csv('./data/Friends_data_only_user_annotation.csv')

# Split into text and labels
texts = data['text'].values
labels = data['label'].values


############## preprocessing ################

# read file with swedish stopwords
# https://github.com/stopwords-iso/stopwords-sv
# https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages
def load_stopwords(file_path):
    words_lst = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for word in file.readlines():
            words_lst.append(word.strip())
    return set(words_lst)

# load stop words
stop_words_file = './data/swedish_stopwords.txt'
stop_words = load_stopwords(stop_words_file)


def preprocess_texts(texts, stop_words):
    processed_texts = []
    
    for text in texts:
        doc = nlp(text)
        tokens = []

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.text.lower() not in stop_words and len(word.text) > 2:
                    tokens.append(word.lemma)
        
        processed_texts.append(" ".join(tokens))

    return processed_texts


