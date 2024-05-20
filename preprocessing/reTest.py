import pandas as pd
import re

# Define a function to check if the text contains alphabetic characters
def contains_alphabetic(text):
    return bool(re.search(r'[a-zA-ZåäöÅÄÖ]', text))

# Create a sample DataFrame similar to what you might have
data = pd.DataFrame({
    'text': [
        'Hello world!',         # Valid English text
        'Hej världen!',         # Valid Swedish text
        '123456',               # Numbers only
        '!!!???',               # Special characters only
        'Grüß Gott!',           # Valid text with special characters
        'hello123',             # Text with numbers
        '... !!!',              # Non-alphabetic characters
        '',                     # Empty string
        '😊😊😊',                 # Emoji only
        'åäö ÅÄÖ',              # Valid Swedish characters
        '   ',                  # Whitespace only
        '123 456 !@#',          # Mixed but no alphabetic characters
        'Correct! #100Times',   # Valid mixed content
        '....se...',
        ": ) : ) : ) :mobbad D !"
    ]
})

# Filter the data to remove rows with only non-alphabetic characters
filtered_data = data[data['text'].apply(contains_alphabetic)]

# Print the original and filtered data for comparison
print("Original Data:\n", data)
print("\nFiltered Data:\n", filtered_data)
