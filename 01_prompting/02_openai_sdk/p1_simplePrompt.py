import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "What is RAG in AI?"

# 2. Call the model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.4,
    messages=[
        {"role": "user", "content": text}
    ]
)

print(response.choices[0].message.content)