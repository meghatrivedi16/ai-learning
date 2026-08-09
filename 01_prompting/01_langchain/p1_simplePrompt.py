import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Initialize model
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.4)

text = "What is RAG in AI?"

prompt = PromptTemplate.from_template(text)

chain = prompt | llm

response = chain.invoke({})

print(response.content)
