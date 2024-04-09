from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

while True:
    # Ask the user for their question
    user_question = input("Enter text:")

    # Check if the user wants to exit
    if user_question.lower() == 'exit':
        break

    completion = client.chat.completions.create(
    model="NousResearch/Hermes-2-Pro-Mistral-7B-GGUF/Hermes-2-Pro-Mistral-7B.Q4_0.gguf",
      messages=[
        {"role": "system", "content": "Respond with just '1!' if the content consist of bullying in school and with just '0!' if it doesn't."},
        {"role": "user", "content": user_question}
      ],
      temperature=0.7,
    )

    print(completion.choices[0].message.content)


