import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


messages = [
    {"role": "system", "content": "Classify sentiment as Positive, Negative, or Mixed."},
    {"role": "user", "content": "Amazing quality, fast shipping!"},
    {"role": "assistant", "content": "Positive"},
    {"role": "user", "content": "Terrible packaging, item arrived broken."},
    {"role": "assistant", "content": "Negative"},
    {"role": "user", "content": "Good product but customer service was slow."},
    {"role": "assistant", "content": "Mixed"},
    {"role": "user", "content": "The delivery was late but the product is great."},
    #{"role": "user", "content": "The product quality is very poor."},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.4,
    messages=messages
)

print(response.choices[0].message.content)