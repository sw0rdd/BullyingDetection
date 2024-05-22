import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
This script is used to find the best threshold for a Support Vector Machine model using 5-fold cross-validation.
The best threshold is then evaluated using the F3-score metric.
"""


# Loading dataset with stop words
data = pd.read_csv('./data/new_preprocessed_friends_data.csv')

# Lading dataset without stop words
# data = pd.read_csv('data/preprocessed_dataset.csv', usecols=['text', 'label'])


data.dropna(subset=['text'], inplace=True)
texts = data['text'].values
labels = data['label'].values

# Setup the imbalanced-learn pipeline with SMOTE
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', SVC(kernel='linear', C=1, gamma='scale', probability=True))
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
thresholds = np.arange(0.01, 0.6, 0.01)
# thresholds = [0.05, 0.01, 0.03, 0.1, 0.2, 0.3]
# thresholds = [0.01, 0.03, 0.1, 0.22, 0.3, 0.5, 0.9]

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


# Threshold 0.03:
#  Class 0 - Precision: 0.98, Recall: 0.70, F1: 0.82, F3: 0.72
#  Class 1 - Precision: 0.40, Recall: 0.94, F1: 0.56, F3: 0.83
#  Average Macro F3 Score: 0.775
# Threshold 0.05:
#  Class 0 - Precision: 0.98, Recall: 0.75, F1: 0.85, F3: 0.77
#  Class 1 - Precision: 0.43, Recall: 0.92, F1: 0.59, F3: 0.83
#  Average Macro F3 Score: 0.796
# Threshold 0.10:
#  Class 0 - Precision: 0.97, Recall: 0.80, F1: 0.88, F3: 0.81
#  Class 1 - Precision: 0.48, Recall: 0.88, F1: 0.62, F3: 0.81
#  Average Macro F3 Score: 0.814
