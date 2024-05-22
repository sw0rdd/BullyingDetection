import pandas as pd
import re

""" 
This script includes the cleaning of the data by removing rows with only non-alphabetic characters.
The cleaned data is then saved to a new CSV file.
"""

# load the data
data = pd.read_csv('./data/Friends_data_only_user_annotation.csv') 

# Drop rows with NaN in 'text' or 'label text' columns
data.dropna(subset=['label text'], inplace=True)

# a function to check for alphabetic characters, including Swedish
def contains_alphabetic(text):
    return bool(re.search(r'[a-zA-ZåäöÅÄÖ]', text))



# Filter the data to remove rows with only non-alphabetic characters
filtered_data = data[data['text'].apply(contains_alphabetic)]

# Save the filtered data
filtered_data.to_csv('./data/Friends_data_only_user_annotation_clean.csv', index=False)



#### Test the function ####

# test the function
# test_data = pd.DataFrame({
#     'text': [
#         'Hello world!',         # Valid English text
#         'Hej världen!',         # Valid Swedish text
#         '123456',               # Numbers only
#         '!!!???',               # Special characters only
#         'Grüß Gott!',           # Valid text with special characters
#         'hello123',             # Text with numbers
#         '... !!!',              # Non-alphabetic characters
#         '',                     # Empty string
#         '😊😊😊',                 # Emoji only
#         'åäö ÅÄÖ',              # Valid Swedish characters
#         '   ',                  # Whitespace only
#         '123 456 !@#',          # Mixed but no alphabetic characters
#         'Correct! #100Times',   # Valid mixed content
#         '....se...',
#         ": ) : ) : ) :mobbad D !"
#     ]
# })

# # print new data
# test_data = test_data[test_data['text'].apply(contains_alphabetic)]
# print(test_data)
