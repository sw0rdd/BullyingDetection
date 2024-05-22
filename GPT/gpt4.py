from openai import OpenAI
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

"""
This script uses the OpenAI API to classify text data using the GPT-4o model.
"""


# Initialize the OpenAI client with API key
client = OpenAI(api_key='####')

# Load the data
data = pd.read_csv('./GPT_Data/Friends_not_sens_filtered.csv')


# Define the cost per token
COST_PER_INPUT_TOKEN = 0.000005  # $5.00 per 1M tokens
COST_PER_OUTPUT_TOKEN = 0.000015  # $15.00 per 1M tokens

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0



def classify_text(prompt, text):
    global total_input_tokens, total_output_tokens, total_cost

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    
    try:
        # Using the updated API method for creating chat completions
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            max_tokens=1,
            temperature=0
        )
    
        # Properly accessing the response content
        label = response.choices[0].message.content.strip()


        # Get token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Calculate cost for this prompt
        input_cost = prompt_tokens * COST_PER_INPUT_TOKEN
        output_cost = completion_tokens * COST_PER_OUTPUT_TOKEN
        cost = input_cost + output_cost
        
        # Update total tokens used and total cost
        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens
        total_cost += cost
        
        # Print cost for this prompt
        print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}")
        print(f"Cost for this prompt: ${cost:.5f}")
        
        return int(float(label))
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# prompt = """Klassificera följande text utifrån om personen som skrivit 
# den verkar mobbad i skolmiljön. Om det tyder på mobbning, svara 
# endast med '1.0'. Annars, svara endast med '0.0'. Klassificeringen 
# gäller enbart om författaren av texten är mobbad. Innehåll som bara 
# är stötande eller våldsamt innebär inte automatiskt att texten ska 
# märkas som '1.0'."""

#new prompt, focusing on fear and threat
prompt = """Klassificera följande text utifrån om personen som skrivit 
den verkar mobbad, rädd, hotad, eller råkar för våld i skolmiljön. Om det tyder på mobbning, 
rädsla eller hot, svara endast med '1.0'. Annars, svara endast med '0.0'. 
Klassificeringen gäller enbart om författaren av texten är mobbad, känner sig 
rädd eller hotad. Innehåll som bara är stötande eller upprörande innebär inte 
automatiskt att texten ska märkas som '1.0'."""


data['predicted_label'] = data['text'].apply(lambda x: classify_text(prompt, x))

# Print total tokens used and total cost
print(f"Total Input Tokens: {total_input_tokens}")
print(f"Total Output Tokens: {total_output_tokens}")
print(f"Total Cost: ${total_cost:.5f}")


# Save the results to a new CSV file
data.to_csv('./GPT_Data/GPT4o_predictions_price.csv', index=False)



# Total Input Tokens: 65080
# Total Output Tokens: 410
# Total Cost: $0.33155
