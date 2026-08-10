import os
from langchain_text_splitters import TokenTextSplitter



sample_text = """Retrieval-Augmented Generation (RAG) is an AI framework that combines a large language model (LLM) with an external information retrieval process. 
In simple terms, RAG-equipped systems search for relevant knowledge (from documents, databases, or the web) before generating answers with the AI model. 

This means the AI isn’t limited to what it “learned” during training – it can pull in fresh, authoritative data on-the-fly. 
The concept was popularized by a 2020 research paper from Facebook (Meta) introducing RAG as a way to give models access to information beyond their training data. 
You can think of it like an open-book exam for the AI, as opposed to a closed-book exam where the AI relies only on memory. 
By allowing lookup of facts in real time, RAG enables more accurate and context-aware responses than models guessing from pre-trained knowledge.

Why does this matter? Modern AI models (like GPT-4 or other LLMs) are extremely powerful at generating text, but they have some known limitations. 
They can “hallucinate” incorrect information or provide outdated answers if asked about recent events or niche topics outside their training data. 
RAG directly tackles these issues by grounding the AI’s responses in up-to-date, factual information retrieved from a knowledge source. In essence, 
RAG lets AI systems augment their own knowledge on demand, making them far more reliable for tasks like question-answering."""


splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)

chunks = splitter.split_text(sample_text)

print(f"Total chunks created: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
