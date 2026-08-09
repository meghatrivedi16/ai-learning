import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

ideas_prompt = ChatPromptTemplate.from_messages([
    ("system", "You generate blog post ideas."),
    ("user", "Give me 3 blog post title ideas about {topic} for beginners.")
])
ideas_chain = ideas_prompt | llm | parser

outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You create structured blog post outlines."),
    ("user", "From these ideas, pick the best one for a beginner audience and create a 5-point outline:\n\n{ideas}")
])
outline_chain = outline_prompt | llm | parser

post_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer who explains AI concepts clearly for beginners."),
    ("user", "Write a 400-word blog post following this outline:\n\n{outline}")
])
post_chain = post_prompt | llm | parser

full_chain = (
    RunnableParallel(ideas=ideas_chain)
    | RunnableLambda(lambda x: {"outline": outline_chain.invoke({"ideas": x["ideas"]})})
    | RunnableLambda(lambda x: {"post": post_chain.invoke({"outline": x["outline"]})})
)

result = full_chain.invoke({"topic": "RAG (Retrieval-Augmented Generation)"})
print(result["post"])