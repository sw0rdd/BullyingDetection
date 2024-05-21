import stanza
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import re


data = pd.read_csv('./data/Friends_data_only_user_annotation.csv') 

# Drop rows with NaN in 'text' or 'label text' columns
data.dropna(subset=['text', 'label'], inplace=True)

# Remove rows with no alphabetic characters in the 'text' column
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