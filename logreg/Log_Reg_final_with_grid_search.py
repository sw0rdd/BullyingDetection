import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script is used to find the best hyperparameters for a logistic regression model using grid search with 5-fold cross-validation.
The best model is then evaluated using the F3-score metric.
"""

# loading dataset
# df = pd.read_csv('data/new_preprocessed_friends_data.csv', usecols=['text', 'label'])

# Lading dataset without stop words
df = pd.read_csv('data/preprocessed_dataset.csv', usecols=['text', 'label'])

# drop rows with missing values
df.dropna(subset=['text'], inplace=True)

# feature extraction
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['text'])
y = df['label']

# resampling strategies
over = SMOTE(random_state=42)

# custom scorer for F3-score label 1
def f3_scorer(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3, pos_label=1)

scorer = make_scorer(f3_scorer)

# create a pipeline that first over-samples, then runs logistic regression
pipeline = Pipeline([
    ('over', over),
    ('model', LogisticRegression(class_weight='balanced'))
])

# specify the grid search parameters
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'model__max_iter': [100, 200, 300],
    'model__penalty': ['l1', 'l2']
}

# using Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# apply grid search to find the best parameters with respect to F3 score
grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=skf, n_jobs=-1, verbose=2) 
grid_search.fit(X_tfidf, y)

# print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# set up the optimized pipeline
best_pipeline = grid_search.best_estimator_

# initialize lists to store metrics across folds
all_probabilities = []
all_true_labels = []

# evaluate the model with optimized parameters using Stratified K-Fold cross-validation
for train_index, test_index in skf.split(X_tfidf, y):
    X_train, X_val = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_val = y[train_index], y[test_index]

    # fit/train the best model
    best_pipeline.fit(X_train, y_train)

    # store probabilities and true labels
    probabilities = best_pipeline.predict_proba(X_val)[:, 1]
    all_probabilities.append(probabilities)
    all_true_labels.append(y_val)

# make the arrays into a 1D array
all_probabilities = np.concatenate(all_probabilities)
all_true_labels = np.concatenate(all_true_labels)

# thresholds to iterate over
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

# initialize variables to store the highest F3-score and corresponding confusion matrix
highest_f3_score = -1
best_conf_matrix = None
best_threshold = None

# evaluate the model at each threshold
for threshold in thresholds:
    predictions = (all_probabilities >= threshold).astype(int)

    # calculate precision, recall, F1, and F3 scores
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, predictions, labels=[0, 1], average=None)
    f3_0 = fbeta_score(all_true_labels, predictions, beta=2, labels=[0], average='binary', pos_label=0)
    f3_1 = fbeta_score(all_true_labels, predictions, beta=3, labels=[1], average='binary', pos_label=1)

    conf_matrix = confusion_matrix(all_true_labels, predictions)

    print(f"\nAverage report after 5-fold cross-validation with threshold {threshold}:")
    print("             labels     precision  recall    f1-score  f3-score")
    print(f"              0.0         {precision[0]:.2f}      {recall[0]:.2f}      {f1[0]:.2f}      {f3_0:.2f}")
    print(f"              1.0        [{precision[1]:.2f}]    [{recall[1]:.2f}]     {f1[1]:.2f}     [{f3_1:.2f}]")

    # find the threshold that gives the highest F3-score for class 1
    if f3_1 > highest_f3_score:
        highest_f3_score = f3_1
        best_conf_matrix = conf_matrix
        best_threshold = threshold

# plot the confusion matrix of the best threshold
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title(f'Confusion Matrix for Threshold {best_threshold} with Highest F3-Score ({highest_f3_score:.2f})')
plt.show()
