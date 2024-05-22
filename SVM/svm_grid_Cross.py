import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, fbeta_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

"""
This script is used to find the best hyperparameters for a Support Vector Machine model using grid search with 5-fold cross-validation.
The best model is then evaluated using the F3-score metric.
"""

# Load data
# data = pd.read_csv('./data/new_preprocessed_friends_data.csv')
data = pd.read_csv('./data/preprocessed_dataset.csv')
data.dropna(subset=['text'], inplace=True)
texts = data['text'].values
labels = data['label'].values

# Setup the imbalanced-learn pipeline with SMOTE
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('classifier', SVC(probability=True))
])

# Define the parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['rbf', 'linear']
}

# Custom scorer for F3 score focusing on the positive class
def f3_score_positive_class(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3, labels=[1], average=None)[0]  # Index 0 for positive class

f3_scorer = make_scorer(f3_score_positive_class)

# Define the F3 score as a custom scorer
# f3_scorer = make_scorer(fbeta_score, beta=3, average='macro')

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f3_scorer, verbose=1, n_jobs=-1)

# Perform the grid search
grid_search.fit(texts, labels)

# Output the best parameters and corresponding score
print("Best parameters found:", grid_search.best_params_)
print("Best F3 score:", grid_search.best_score_)


# Best parameters found: {'classifier__C': 1, 'classifier__gamma': 'scale', 'classifier__kernel': 'linear'}
# Best F3 score: 0.7464196321798788
