import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script is used to find the best threshold for a Multinomial Naive Bayes model using 5-fold cross-validation.
The best threshold is then evaluated using the F3-score metric.
"""

# Load data with stop words
# data = pd.read_csv('./data/new_preprocessed_friends_data.csv')

# Lading dataset without stop words
data = pd.read_csv('data/preprocessed_dataset.csv', usecols=['text', 'label'])


data.dropna(subset=['text'], inplace=True)
texts = data['text'].values
labels = data['label'].values

# Setup the imbalanced-learn pipeline with SMOTE
pipeline = ImbPipeline([
    ('vectorizer', CountVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', MultinomialNB(alpha=0.16))
])


# Setup cross-validation with stratified k-fold (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store all probabilities and true labels across folds
all_probabilities = []
all_true_labels = []

# Cross-validation loop
for train_index, test_index in cv.split(texts, labels):
    X_train, X_test = texts[train_index], texts[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Fit data using pipeline
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Store probabilities and true labels for overall evaluation
    all_probabilities.extend(probabilities)
    all_true_labels.extend(y_test)

# Convert lists to arrays for threshold evaluation
all_probabilities = np.array(all_probabilities)
all_true_labels = np.array(all_true_labels)

# Initialize variables to store the highest F3-score and corresponding confusion matrix
highest_f3_score = -1
best_conf_matrix = None
best_threshold = None

# defining thresholds to explore
# thresholds = np.linspace(0.01, 0.3, 30)
# thresholds = np.arange(0.01, 1, 0.01)
thresholds = [0.01, 0.1, 0.22, 0.23, 0.3, 0.5, 0.9]



# initialize dictionary to store metrics for each threshold
threshold_metrics = {th: {'precision': [], 'recall': [], 'f1': [], 'f3': [], 'macro_f3': []} for th in thresholds}


# Evaluate the model at each threshold
for threshold in thresholds:
    predictions = (all_probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, predictions, labels=[0, 1], average=None)
    f3_scores = fbeta_score(all_true_labels, predictions, beta=3, labels=[0, 1], average=None)
    f3_overall = fbeta_score(all_true_labels, predictions, beta=3, average='macro')


    conf_matrix = confusion_matrix(all_true_labels, predictions)

    # Print detailed metrics for each threshold
    print(f"\nThreshold: {threshold:.2f}")
    print("Class 0 - Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, F3-Score: {:.2f}".format(precision[0], recall[0], f1[0], f3_scores[0]))
    print("Class 1 - Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, F3-Score: {:.2f}".format(precision[1], recall[1], f1[1], f3_scores[1]))
    print(f"Macro F3-Score: {f3_overall:.2f}")


    # Check if this threshold gives the highest F3-score for class 1 and update best scores and matrix
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
plt.savefig('./img/finals/mnbFINAL.png')
# plt.show()  

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