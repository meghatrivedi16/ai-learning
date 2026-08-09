import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

# 1. Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the shape you want back — this IS the schema, no separate JSON dict needed
class Sentiment(str, Enum):
    positive = "Positive"
    negative = "Negative"
    mixed = "Mixed"

class SentimentResult(BaseModel):
    sentiment: Sentiment = Field(description="Overall sentiment of the text")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation for the classification")

# Note: .parse() instead of .create(), and response_format=SentimentResult
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0.4,
    messages=[
        {"role": "system", "content": "Classify the sentiment of the given text."},
        {"role": "user", "content": "The delivery was late but the product is great."}
    ],
    response_format=SentimentResult   # ← this is the missing piece
)

result = response.choices[0].message.parsed   # already a SentimentResult object

print(result)                  # sentiment=<Sentiment.mixed: 'Mixed'> confidence=0.7 reasoning='...'
print(result.sentiment)        # Sentiment.mixed
print(result.sentiment.value)  # "Mixed"
print(result.confidence)       # 0.7
print(result.reasoning)        # "..."

