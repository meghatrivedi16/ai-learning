"""
LAB: Module 2 - Simple RAG Query Program
---------------------------------------------------------
Goal: Query the ingested VelocityX Policy from ChromaDB.

This program:
1. Connects to the existing ChromaDB vector store.
2. Retrieves relevant policy chunks.
3. Sends the retrieved context to the LLM.
4. Runs built-in test cases.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -----------------------------
# 1. CONFIGURATION
# -----------------------------

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "velocity_x_policy"


# -----------------------------
# 2. LOAD VECTOR DATABASE
# -----------------------------

def load_vector_db():
    """
    Loads the existing ChromaDB collection created by the ingestion pipeline.
    """

    embeddings = OpenAIEmbeddings()

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    return vector_db


# -----------------------------
# 3. FORMAT RETRIEVED DOCUMENTS
# -----------------------------

def format_docs(docs):
    """
    Converts retrieved chunks into a single context string.
    """

    formatted_chunks = []

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "unknown")

        formatted_chunks.append(
            f"\n--- Retrieved Chunk {i} | Page: {page} ---\n"
            f"{doc.page_content}"
        )

    return "\n".join(formatted_chunks)


# -----------------------------
# 4. CREATE RAG CHAIN
# -----------------------------

def create_rag_chain(vector_db):
    """
    Creates a simple Retrieval-Augmented Generation chain.
    """

     #
     # This line just builds an object and says: "whenever you're later given a query, do a similarity search and return the top 4 matches." 
     # No question exists yet. No search happens yet.
     #

     #
     # A retriever is a standard interface/component in LangChain whose job is exactly one thing: given a query string, 
     # return the most relevant documents/chunks from some data source.
     # That's it. It doesn't generate text, doesn't call an LLM, doesn't know about prompts. It's purely a search step.
     #

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a customer service policy assistant for VelocityX.

Use ONLY the provided policy context to answer the question.
If the answer is not available in the context, say:
"I could not find this in the provided VelocityX policy."

Give a clear, practical answer that a customer service agent can use.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# -----------------------------
# 5. ASK ONE QUESTION
# -----------------------------

def ask_question(question, rag_chain, retriever, show_sources=True):
    """
    Runs a single question through the RAG chain.
    Optionally prints retrieved source chunks for diagnostics.
    """

    print("\n==================================================")
    print(f"QUESTION: {question}")
    print("==================================================")

    if show_sources:
        docs = retriever.invoke(question)

        print("\n--- Retrieved Source Chunks ---")
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "unknown")
            preview = doc.page_content[:500].replace("\n", " ")

            print(f"\nSource {i}")
            print(f"Page: {page}")
            print(f"Preview: {preview}...")

    answer = rag_chain.invoke(question)

    print("\n--- RAG Answer ---")
    print(answer)

    return answer


# -----------------------------
# 6. TEST CASES
# -----------------------------

def run_test_cases(rag_chain, retriever):
    """
    Runs sample questions against the VelocityX policy.
    These test cases cover returns, exchanges, warranty,
    DOA, refunds, and escalation.
    """

    test_questions = [
        "What is the standard return window for VelocityX ecommerce purchases?",

        "A customer says the sole came off after three weeks of normal use. Is this covered under warranty?",

        "Can a customer return opened socks if they simply changed their mind?",

        "What should an agent do if the customer received size 8 but ordered size 9?",

        "What evidence is required for a damaged-on-arrival claim?",

        "How long does a typical refund take after warehouse quality check?",

        "Are customized products returnable if the customer does not like the fit?",

        "What should the agent do if a customer is outside the 30-day return window by a few days?",

        "What are some fraud or abuse indicators in return claims?",

        "When should a case be escalated to Legal or Compliance?"
    ]

    for question in test_questions:
        ask_question(
            question=question,
            rag_chain=rag_chain,
            retriever=retriever,
            show_sources=True
        )


# -----------------------------
# 7. MAIN PROGRAM
# -----------------------------

if __name__ == "__main__":
    print("Loading VelocityX Policy vector database...")

    vector_db = load_vector_db()

    print("Vector database loaded successfully.")

    rag_chain, retriever = create_rag_chain(vector_db)

    print("RAG chain created successfully.")

    run_test_cases(rag_chain, retriever)