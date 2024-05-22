import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch


"""
This script is used to translate text data from English to Swedish using the MarianMT model.
The translated data is then saved to a new CSV file.
"""


# Function to translate text using the MarianMT model
def translate_texts(texts, model_name="Helsinki-NLP/opus-mt-en-sv"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Batch translation to manage memory usage
    batch_size = 16
    translated_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = model.generate(**inputs)
            translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    
    return translated_texts

# Load datasets
aggressive_data = pd.read_csv('./data/Aggressive_All.csv')
non_aggressive_data = pd.read_csv('./data/Non_Aggressive_All.csv')

# Select 50k samples from each
aggressive_samples = aggressive_data['Message'].iloc[:50000]
non_aggressive_samples = non_aggressive_data['Message'].iloc[:50000]

# Translate texts
translated_aggressive = translate_texts(aggressive_samples.tolist())
translated_non_aggressive = translate_texts(non_aggressive_samples.tolist())

# Create a new DataFrame
new_data = pd.DataFrame({
    'text': translated_aggressive + translated_non_aggressive,
    'label': [1] * len(translated_aggressive) + [0] * len(translated_non_aggressive)
})

# Save to new CSV file
new_data.to_csv('translated_dataset.csv', index=False)

print("Translation complete and data saved.")


# # Custom array of three sentences for testing
# test_sentences = [
#     "This is a test sentence.",
#     "How does the translation handle this?",
#     "Just another example to check the output."
# ]

# # Call the translate function
# translated_sentences = translate_texts(test_sentences)
# print(translated_sentences)