from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("AI-Nordics/bert-large-swedish-cased")
model = AutoModelForMaskedLM.from_pretrained("AI-Nordics/bert-large-swedish-cased")


inputs = tokenizer("Your text here", return_tensors="pt")


outputs = model(**inputs)
