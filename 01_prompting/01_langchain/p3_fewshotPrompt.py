import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

examples = [
    {"text": "Amazing quality, fast shipping!", "sentiment": "Positive"},
    {"text": "Terrible packaging, item arrived broken.", "sentiment": "Negative"},
    {"text": "Good product but customer service was slow.", "sentiment": "Mixed"},
]

example_prompt = PromptTemplate.from_template(
    "Text: {text}\nSentiment: {sentiment}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment as Positive, Negative, or Mixed.",
    suffix="Text: {input}\nSentiment:",
    input_variables=["input"],
)

chain = few_shot_prompt | llm
response = chain.invoke({"input": "The delivery was late but the product is great."})
print(response.content)