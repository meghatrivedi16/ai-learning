import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate.from_template(
    "Explain {topic} in {language} for a {audience}."
)

chain = prompt | model

response = chain.invoke({
    "topic": "RAG",
    "language": "English",
    "audience": "Java developer"
})

print(response.content)