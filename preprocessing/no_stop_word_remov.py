import stanza
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import re

"""
This script include the preprocessing of the data using the stanza library.
It performs tokenization and lemmatization, stop words are not removed.
The preprocessed data is then saved to a new CSV file.
"""

############# setup and data loading ################

# Ensure the Swedish model is downloaded
stanza.download('sv', verbose=False)

# Initialize the Stanza pipeline for Swedish, without the mwt processor
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma', verbose=False, use_gpu=True)

data = pd.read_csv('./data/translated_dataset.csv')

# Drop rows with NaN in 'text' or 'label text' columns
data.dropna(subset=['text', 'label'], inplace=True)

data = data[data['text'].apply(lambda x: bool(re.search(r'[a-zA-ZåäöÅÄÖ]', x)))]

# Split into text and labels
texts = data['text'].values
labels = data['label'].values


############## preprocessing ################


def preprocess_texts(texts):
    processed_texts = []
    
    for text in texts:
        doc = nlp(text)
        tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
        processed_texts.append(" ".join(tokens))

    return processed_texts


# Process texts
preprocessed_texts = preprocess_texts(texts)

# Combine processed texts with labels
preprocessed_data = pd.DataFrame({
    'text': preprocessed_texts,
    'label': labels
})

# Save the preprocessed data to a new CSV file
preprocessed_data.to_csv('./new_preprocessed_trans_withStopWords.csv', index=False)

print("Preprocessed data saved to 'new_preprocessed_trans_withStopWords.csv'.")