import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Define the shape you want back — this IS the schema, no separate JSON dict needed
class Sentiment(str, Enum):
    positive = "Positive"
    negative = "Negative"
    mixed = "Mixed"

class SentimentResult(BaseModel):
    sentiment: Sentiment = Field(description="Overall sentiment of the text")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation for the classification")

# 2. Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# 3. Bind the schema to the model — this is the key LangChain-specific step
structured_llm = llm.with_structured_output(SentimentResult)

# 4. Build the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment of the given text."),
    ("user", "{text}")
])

# 5. Chain it together
chain = prompt | structured_llm

# 6. Invoke
result = chain.invoke({"text": "The delivery was late but the product is great."})

print(result)                  # SentimentResult(sentiment=<Sentiment.mixed>, confidence=0.7, reasoning='...')
print(result.sentiment)        # Sentiment.mixed
print(result.sentiment.value)  # "Mixed"
print(result.confidence)       # 0.7
print(result.reasoning)        # "..."
print(type(result))            # <class '__main__.SentimentResult'>