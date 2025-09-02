import os

from groq import Groq

client = Groq(
    # This is the default and can be omitted
    api_key="api",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="openai/gpt-oss-20b",
)

llama-3.1-8b-instant
llama-3.3-70b-versatile
openai/gpt-oss-20b

print(chat_completion.choices[0].message.content)
