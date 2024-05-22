import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

"""
This script is used to find the best hyperparameters for a Multinomial Naive Bayes model using grid search with 5-fold cross-validation.
The best model is then evaluated using the F3-score metric.
"""

# Load data
# data = pd.read_csv('./data/new_preprocessed_friends_data.csv')

# Lading dataset without stop words
data = pd.read_csv('data/preprocessed_dataset.csv', usecols=['text', 'label'])

data.dropna(subset=['text'], inplace=True)
texts = data['text'].values
labels = data['label'].values

# Setting up the pipeline and grid search
pipeline = ImbPipeline([
    ('vect', CountVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', MultinomialNB())])

params = {
    'clf__alpha': [0, 0.01, 0.1, 0.165, 0.5, 1, 2] 
    # 'clf__alpha': np.linspace(0.01, 1, 20) 
    # 'clf__alpha': [0, 0.165, 0.13, 0.14] 
    # 'clf__alpha': np.arange(0.01, 1, 0.01)
}

cross_val = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Custom scorer for F3 score focusing on the positive class
def f3_score_positive_class(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3, labels=[1], average=None)[0]  # Index 0 for positive class

f3_scorer = make_scorer(f3_score_positive_class)

grid_search = GridSearchCV(pipeline, param_grid=params, scoring=f3_scorer, cv=cross_val, n_jobs=-1, verbose=1)
grid_search.fit(texts, labels)

print("Best alpha:", grid_search.best_params_)
print("Best cross-validation F3 score for positive class:", grid_search.best_score_)