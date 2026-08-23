"""
LAB: Module 2 - Advanced RAG Pipeline
---------------------------------------------------------
Advanced Concepts:
1. HyDE: Generates a 'hypothetical' answer to improve semantic matching.
2. RRF (Reciprocal Rank Fusion): Merges results from different retrieval paths.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever


load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
embeddings = OpenAIEmbeddings()

# Load the existing Vector Store
vector_db = Chroma(
    collection_name="velocity_x_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# --- ADVANCED RETRIEVAL COMPONENTS ---

def get_hyde_query(original_query):
    """HyDE: Generate a hypothetical answer to use as a search vector."""
    prompt = f"Write a brief sample policy answer for this customer query: {original_query}"
    # Use .invoke() instead of .predict()
    # .content is needed because .invoke() returns a BaseMessage object
    response = llm.invoke(prompt) 
    print(f"HyDE Context: {response.content}")  # Debug: See the generated context
    return response.content

# --- HYBRID SEARCH & RRF IMPLEMENTATION ---

def run_rag_pipeline(user_query):
    print(f"\n--- PROCESSING: {user_query} ---")

    # 1. Initialize BM25 (Keyword Search) 
    # In production, you'd fetch all docs from Chroma to build this retriever.
    all_docs = vector_db.get()['documents']
    bm25_retriever = BM25Retriever.from_texts(all_docs)   # creating a BM25 retriever using the documents from the vector database. This allows us to perform keyword-based retrieval on the same set of documents that are indexed in the vector store.
    bm25_retriever.k = 3

    # 2. Semantic Search (Vector) 
    semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. Ensemble Retrieval (Hybrid Search with RRF) 
    # EnsembleRetriever uses RRF by default to combine keyword and semantic results.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6] # Weighting semantic search slightly higher
    )

    # 4. Execute Retrieval using HyDE for enrichment 
    hyde_context = get_hyde_query(user_query)
    #docs = ensemble_retriever.get_relevant_documents(hyde_context)
    docs = ensemble_retriever.invoke(hyde_context)

    # 5. Final Generation
    context_text = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = f"Using this policy context (strictly from the provided text):\n{context_text}\n\nAnswer the question: {user_query}"
    
    #response = llm.predict(final_prompt)
    response = llm.invoke(final_prompt).content
    return response

# --- TEST CASES FOR COMPLEX SCENARIOS ---
if __name__ == "__main__":
    test_cases = [
        "I used my football boots on concrete and they tore. Am I covered?", # Logic: Design intent [cite: 340, 383, 782]
        "My order ORD-202 has a sole coming off after 3 weeks. What do I do?", # Logic: Manufacturing defect [cite: 306, 773, 774]
        "I lost my receipt but I am a loyal customer. Can I return my unused shoes?", # Logic: Exception handling [cite: 283, 361, 745]
        "What is the warranty for a pair of running shoes in India?" # Logic: Regional vs Category rules [cite: 310, 443]
    ]

    for query in test_cases:
        answer = run_rag_pipeline(query)
        print(f"Response: {answer}\n")