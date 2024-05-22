import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, fbeta_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script is used to find the best threshold for a logistic regression model using 5-fold cross-validation.
The best threshold is then evaluated using the F3-score metric.
"""


# Loading dataset with stop words
# df = pd.read_csv('data/new_preprocessed_friends_data.csv', usecols=['text', 'label'])

# Lading dataset without stop words
df = pd.read_csv('data/preprocessed_dataset.csv', usecols=['text', 'label'])

# Drop rows with NaN in 'text' or 'label text' columns
df.dropna(subset=['text', 'label'], inplace=True)

# Extract text and labels
texts = df['text']
labels = df['label']

# Setup the imbalanced-learn pipeline with SMOTE
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(class_weight='balanced'))
])

# Setup cross-validation with stratified k-fold (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store all probabilities and true labels across folds
all_probabilities = []
all_true_labels = []

# Cross-validation loop
for train_index, test_index in skf.split(texts, labels):
    X_train, X_val = texts.iloc[train_index], texts.iloc[test_index]
    y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]

    # Fit data using pipeline
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    # Store probabilities and true labels for overall evaluation
    all_probabilities.append(probabilities)
    all_true_labels.append(y_val)

# Convert lists to arrays for threshold evaluation
all_probabilities = np.concatenate(all_probabilities)
all_true_labels = np.concatenate(all_true_labels)


# Thresholds to iterate over
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
# thresholds = [0.01, 0.1, 0.25, 0.3, 0.5, 0.9]


# Initialize variables to store the highest F3-score and corresponding confusion matrix
highest_f3_score = -1
best_conf_matrix = None
best_threshold = None

# Evaluate the model at each threshold
for threshold in thresholds:
    predictions = (all_probabilities >= threshold).astype(int) 

    # Calculates precision, recall, and F1/F3 scores.
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, predictions, labels=[0, 1], average=None)
    f3_scores = fbeta_score(all_true_labels, predictions, beta=3, labels=[0, 1], average=None)
    f3_overall = fbeta_score(all_true_labels, predictions, beta=3, average='macro')

    conf_matrix = confusion_matrix(all_true_labels, predictions)

    # Print detailed metrics for each threshold
    print(f"\nAverage report after 5-fold cross-validation with threshold {threshold}:")
    print("             labels     precision  recall    f1-score  f3-score")
    print(f"              0.0         {precision[0]:.2f}      {recall[0]:.2f}      {f1[0]:.2f}      {f3_scores[0]:.2f}")
    print(f"              1.0        [{precision[1]:.2f}]    [{recall[1]:.2f}]     {f1[1]:.2f}     [{f3_scores[1]:.2f}]")
    print(f"Overall F3 Score: {f3_overall:.3f}")

    # Find the threshold that gives the highest F3-score for class 1
    if f3_scores[1] > highest_f3_score:
        highest_f3_score = f3_scores[1]
        best_conf_matrix = conf_matrix
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.2f}")
print(f"Best F3-Score for class 1: {highest_f3_score:.2f}")
print(best_conf_matrix)


# plot the confusion matrix of the best threshold
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title(f'Confusion Matrix for Threshold {best_threshold} with Highest F3-Score ({highest_f3_score:.2f})')
plt.savefig('./img/finals/logregFINAL.png')


# Plot histograms for the combined probabilities
y_true_0 = all_true_labels == 0
y_true_1 = all_true_labels == 1

plt.figure()
plt.hist(all_probabilities[y_true_0], bins=80, alpha=0.5, label='Non-bullying (label 0)', color='blue')
plt.hist(all_probabilities[y_true_1], bins=80, alpha=0.5, label='Bullying (label 1)', color='red')
plt.title('Combined Distribution of Predicted Probabilities')
plt.xlabel('Probability of Positive Class')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()  