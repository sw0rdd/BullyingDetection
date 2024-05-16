import pandas as pd
from sklearn.model_selection import train_test_split
import chromadb
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

# loading friends data. this dataset is already cleaned from unessesary colunms and nan values
data = pd.read_csv('clean_friends.csv')

# "only use 1 procent of the data fast debugging
# data = data.sample(frac=0.01, random_state=42)

# ensure all data['text'] are strings
data['text'] = data['text'].astype(str)

# split the data into two parts: one for the vector database and one for testing
data_vector_db, data_rag_test = train_test_split(data, test_size=0.3, random_state=42)

# initialize Chroma client and create a collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="friends_collection")

# prepare the metadata/labels
labels = []
for label in data_vector_db['label']:
    labels.append({'label': label})

# add text documents to the collection with metadata and ids
collection.add(
    documents=data_vector_db['text'].tolist(), # e.g. ['jag blir mobbad', ... , 'klassen är stökig.']
    metadatas=labels, # e.g. [{'label': 0.0}, {'label': 1.0}, ... , {'label': 1.0}]
    ids=data_vector_db.index.astype(str).tolist() #  e.g.  ['3614', '2325', '1032', '2323']
)




###################
## prompting LLM ##
###################
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# to store predictions and actual labels
predictions = []
actual_labels = []

# query each text from data_rag_test and classify
for count, (index, row) in enumerate(data_rag_test.iterrows(), 1):
    print('\n' * 10 + '-' * 300)
    print(f" Processing {count}/{len(data_rag_test)}...")  # indicates progress
   
    text_to_classify = row['text']
    actual_labels.append(row['label'])
   
    # querying the collection for the most similar 100 texts
    most_similar_collection = collection.query(
        query_texts=[text_to_classify],
        n_results=100

    )

    # extract texts, labels, and distances
    texts = most_similar_collection['documents'][0] 
    labels = most_similar_collection['metadatas'][0]
    
    # organize results by label, collect the first 3 closest matches for each label
    results_by_label = {1: [], 0: []}
    for i in range(len(texts)):
        label = labels[i]['label']
        if len(results_by_label[label]) < 3:
            results_by_label[label].append(texts[i])
            if len(results_by_label[0]) == 3 and len(results_by_label[1]) == 3:
                break  # stop once we have 3 texts for each label

    # prepare a string of example texts for the prompt
    similar_texts_string = ""
    for label in results_by_label:
        for text in results_by_label[label]:
            similar_texts_string += f"- {text}, label: {label}\n"

    # make RAG prediction
    valid_response = False
    while not valid_response:

        system_prompt = f"""\nKlassificera den sista textmeningen utifrån om personen som skrivit den verkar mobbad i skolmiljön. Om det tyder på mobbning, svara endast med '1.0'. Annars, svara endast med '0.0'. Klassificeringen gäller enbart om författaren av texten är mobbad. Innehåll som bara är stötande eller våldsamt innebär inte automatiskt att texten ska märkas som '1.0'.\n\nHär är sex märkta exempel:\n{similar_texts_string}\n\nKlassifiera texten i user prompten: """
        user_prompt = f"{text_to_classify}\n\n(OBS! svara endast '1.0' eller '0.0', inget annat!)"
        print("\nPROMPT TO LLAMA 3:\n" + '- ' * 300 + system_prompt + user_prompt + '\n' + '- ' * 300)
            
        completion = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        print("\n - LLama response:  " + completion.choices[0].message.content)

        response = completion.choices[0].message.content.strip()
        if response == '1.0' or response == '0.0':
            valid_response = True
            predicted_label = float(response)
            predictions.append(predicted_label)
            actual_label = row['label']
            print(" - Actual Label: ", actual_label)  
        
        print('-' * 300)




########################
## evaluate the model ##
########################
report = classification_report(actual_labels, predictions, target_names=['Not Bullying', '    Bullying'])
f3_score_bullying = fbeta_score(actual_labels, predictions, beta=3, pos_label=1)
f3_score_non_bullying = fbeta_score(actual_labels, predictions, beta=3, pos_label=0)
f3_score_macro = (f3_score_bullying + f3_score_non_bullying) / 2

print(report)
print(f"F3 Score for Bullying: {f3_score_bullying}")
print(f"F3 Score for Non-Bullying: {f3_score_non_bullying}")
print(f"F3 Score Macro: {f3_score_macro}")

# evaluate the model using a confusion matrix
conf_matrix = confusion_matrix(actual_labels, predictions)
tn, fp, fn, tp = conf_matrix.ravel()

# less confusing explanation
print(f"\n  {tn + fp + fn + tp} cases in total in test set.")
print(f"\n  {tp} out of {tp + fn} actual bullying cases identified (true Positives).")
print(f"  {fp} false positives (non-bullying identified as bullying).")
print(f"  {fn} bullying cases missed (false negatives).")
print(f"  {tn} correctly identified non-bullying cases (true negatives).")


# plot the confusion matrix of the best threshold
sns.set_theme(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title(f'Confusion Matrix, F3-Score ({f3_score_bullying:.2f})')
