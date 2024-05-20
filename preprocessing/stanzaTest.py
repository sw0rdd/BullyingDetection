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
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma', verbose=False, use_gpu=True)

data = pd.read_csv('./data/translated_dataset.csv')

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
                    tokens.append(word.text.lower())
        
        processed_texts.append(" ".join(tokens))

    return processed_texts


swedish_sentences = [
    "Biblioteket har böcker man kan låna.",
    "Vädret var perfekt för en lång promenad idag.",
    "Detta är ett meddelande som innehåller mobning",
    "bo"
]



# Example usage with a sample of texts
tokenized_texts = preprocess_texts(swedish_sentences, stop_words)

for text in tokenized_texts:
    print(text)


