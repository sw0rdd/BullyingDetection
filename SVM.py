import pandas as pd 
import re 
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import binarize

# Load your dataset
# df = pd.read_csv('Friends_data_only_user_annotation.csv', usecols=['text', 'label text'])
df = pd.read_csv('./En kolumn elevenkät 6-9_2023-03-15.csv', usecols=['text', 'label text']) # only use the text and label text columns

# Preprocess the text
def preprocess_text(text):
    # Convert to string in case there are non-string types
    text = str(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the preprocessing function to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Drop rows with any NaN values in 'text' or 'label text'
df.dropna(subset=['text', 'label text'], inplace=True)

# Ensure 'label text' is of type int
df['label text'] = df['label text'].astype(int)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label text'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorization to convert text data into numerical data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Model selection and training with class_weight parameter
svmModel = SVC(random_state=42, class_weight='balanced')  # Adjust the class weight here
svmModel.fit(X_train_tfidf, y_train)

# Predict probabilities
y_pred = svmModel.predict(X_val_tfidf)

# Evaluation

print(classification_report(y_val, y_pred))
