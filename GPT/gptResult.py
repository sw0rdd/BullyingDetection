import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, precision_recall_fscore_support

"""
This code snippet is used to evaluate the performance of the GPT-4o model on a classification task.
"""

actual_data = pd.read_csv('./GPT_Data/Friends_not_sens_filtered.csv')

first_prompt_data = pd.read_csv('./GPT_Data/GPT4o_predictions_1.csv')

second_prompt_data = pd.read_csv('./GPT_Data/GPT4o_predictions_2.csv')


first_prompt = """Klassificera följande text utifrån om personen som skrivit 
den verkar mobbad i skolmiljön. Om det tyder på mobbning, svara 
endast med '1.0'. Annars, svara endast med '0.0'. Klassificeringen 
gäller enbart om författaren av texten är mobbad. Innehåll som bara 
är stötande eller våldsamt innebär inte automatiskt att texten ska 
märkas som '1.0'."""

#new prompt, including focus on fear and threat
second_prompt = """Klassificera följande text utifrån om personen som skrivit 
den verkar mobbad, rädd, hotad, eller råkar för våld i skolmiljön. Om det tyder på mobbning, 
rädsla eller hot, svara endast med '1.0'. Annars, svara endast med '0.0'. 
Klassificeringen gäller enbart om författaren av texten är mobbad, känner sig 
rädd eller hotad. Innehåll som bara är stötande eller upprörande innebär inte 
automatiskt att texten ska märkas som '1.0'."""


# counting the actual data
counts = actual_data['label text'].value_counts()
count_0 = counts[0.0]
count_1 = counts[1.0]
print(f"Actual non-bullying instances ===> {count_0}")
print(f"Actual bullying insances ===> {count_1}")
print()

# print("====== First Prompt ======", end='\n\n')
# print(first_prompt, end='\n\n')

print("====== Second Prompt ======", end='\n\n')
print(second_prompt, end='\n\n')

y_true = actual_data['label text']
y_pred = second_prompt_data['predicted_label']
# y_pred = second_prompt_data['predicted_label']


precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None)
f3_scores = fbeta_score(y_true, y_pred, beta=3, labels=[0, 1], average=None)
f3_overall = fbeta_score(y_true, y_pred, beta=3, average='macro')

print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print("Class 0 - Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, F3-Score: {:.2f}".format(precision[0], recall[0], f1[0], f3_scores[0]))
print("Class 1 - Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, F3-Score: {:.2f}".format(precision[1], recall[1], f1[1], f3_scores[1]))
print(f"Macro F3-Score: {f3_overall:.2f}")
print()

conf_matrix = confusion_matrix(y_true, y_pred)
print(f"True Positive: {conf_matrix[1][1]}")
print(f"True Negative: {conf_matrix[0][0]}")
print(f"False Positive: {conf_matrix[0][1]}")
print(f"False Negative: {conf_matrix[1][0]}")







# plot the confusion matrix of the best threshold
# sns.set(style='whitegrid')
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted Labels')
# plt.ylabel('Actual Labels')
# plt.title(f'Confusion Matrix for Threshold {best_threshold} with Highest F3-Score ({highest_f3_score:.2f})')
# plt.savefig('./img/finals/svmFINAL.png')
# plt.show()  


# =============== OUTPUT ===============

# Actual non-bullying instances ===> 210
# Actual bullying insances ===> 200

# ====== First Prompt ======

# Klassificera följande text utifrån om personen som skrivit
# den verkar mobbad i skolmiljön. Om det tyder på mobbning, svara
# endast med '1.0'. Annars, svara endast med '0.0'. Klassificeringen
# gäller enbart om författaren av texten är mobbad. Innehåll som bara
# är stötande eller våldsamt innebär inte automatiskt att texten ska
# märkas som '1.0'.

# Accuracy: 0.8195121951219512
# Class 0 - Precision: 0.85, Recall: 0.79, F1-Score: 0.82, F3-Score: 0.80
# Class 1 - Precision: 0.79, Recall: 0.85, F1-Score: 0.82, F3-Score: 0.84
# Macro F3-Score: 0.82

# True Positive: 170
# True Negative: 166
# False Positive: 44
# False Negative: 30

# ____________________________________________________________

# Actual non-bullying instances ===> 210
# Actual bullying insances ===> 200

# ====== Second Prompt ======

# Klassificera följande text utifrån om personen som skrivit
# den verkar mobbad, rädd, hotad, eller råkar för våld i skolmiljön. Om det tyder på mobbning,
# rädsla eller hot, svara endast med '1.0'. Annars, svara endast med '0.0'.
# Klassificeringen gäller enbart om författaren av texten är mobbad, känner sig
# rädd eller hotad. Innehåll som bara är stötande eller upprörande innebär inte
# automatiskt att texten ska märkas som '1.0'.

# Accuracy: 0.8170731707317073
# Class 0 - Precision: 0.93, Recall: 0.70, F1-Score: 0.80, F3-Score: 0.71
# Class 1 - Precision: 0.75, Recall: 0.94, F1-Score: 0.83, F3-Score: 0.92
# Macro F3-Score: 0.82

# True Positive: 189
# True Negative: 146
# False Positive: 64
# False Negative: 11
