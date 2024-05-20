import stanza
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import time


############# setup and data loading ################

# Ensure the Swedish model is downloaded
stanza.download('sv', verbose=False)


# Initialize the Stanza pipeline for Swedish, without the mwt processor
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma', verbose=False, use_gpu=False)

# https://huggingface.co/stanfordnlp/stanza-sv

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

def preprocess_texts_two(texts, stop_words):
    processed_texts = []
    
    for text in texts:
        doc = nlp(text)
        tokens = [word.lemma for sentence in doc.sentences for word in sentence.words if len(word.text) > 2 and word.text.lower() not in stop_words] 
        processed_texts.append(" ".join(tokens))

    return processed_texts


example_texts = [
    "Det var en gång en katt som hette Nils."
]

print(preprocess_texts(example_texts, stop_words))
print(preprocess_texts_two(example_texts, stop_words))