"""
LAB: Module 2 - Multi-Query RAG Pipeline
---------------------------------------------------------
Advanced Concept:
1. Multi-Query Retrieval: Generates multiple versions of the user question to improve retrieval recall.

It uses:
- ChromaDB vector search
- OpenAI embeddings
- ChatOpenAI for query expansion and answer generation
"""

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma


load_dotenv()


# -----------------------------
# 1. INITIALIZE LLM + EMBEDDINGS
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3
)

embeddings = OpenAIEmbeddings()


# -----------------------------
# 2. LOAD EXISTING CHROMA DB
# -----------------------------

vector_db = Chroma(
    collection_name="velocity_x_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# -----------------------------
# 3. GENERATE MULTI-QUERY VARIATIONS
# -----------------------------

def get_multi_queries(original_query):
    """
    Generates alternative search queries from the original user question.

    The purpose is to improve retrieval recall.
    Different query phrasings may match different chunks in the vector DB.
    """

    prompt = f"""
You are helping improve retrieval for a RAG system.

Generate 3 alternative search queries for the user question below.

Rules:
- Preserve all key facts from the original question.
- Return only the queries.
- Do not add explanations.
- Each query should be on a new line.
- Keep each query short and search-focused.

User question:
{original_query}
"""

    response = llm.invoke(prompt).content

    queries = [
        line.strip()
        for line in response.split("\n")
        if line.strip()
    ]

    print("\n--- Multi-Query Variations ---")
    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")

    return queries


# -----------------------------
# 4. RETRIEVE DOCUMENTS
# -----------------------------

def retrieve_with_multi_query(user_query, k=3):
    """
    Runs vector search for the original query and all generated query variations.
    Then deduplicates the retrieved chunks.
    """

    # Generate alternative query versions
    generated_queries = get_multi_queries(user_query)

    # Include the original query also
    all_queries = [user_query] + generated_queries

    all_retrieved_docs = []

    print("\n--- Retrieval Diagnostics ---")

    for query in all_queries:
        print(f"\nRetrieving for query: {query}")

        retrieved_docs = vector_db.similarity_search(
            query=query,
            k=k
        )

        for i, doc in enumerate(retrieved_docs, start=1):
            page = doc.metadata.get("page", "unknown")
            preview = doc.page_content[:250].replace("\n", " ")

            print(f"  Chunk {i} | Page: {page}")
            print(f"  Preview: {preview}...")

        all_retrieved_docs.extend(retrieved_docs)

    # Deduplicate chunks using page_content as the key
    unique_docs = {}

    for doc in all_retrieved_docs:
        key = doc.page_content
        unique_docs[key] = doc

    final_docs = list(unique_docs.values())

    print(f"\nTotal retrieved chunks before deduplication: {len(all_retrieved_docs)}")
    print(f"Total unique chunks after deduplication: {len(final_docs)}")

    return final_docs


# -----------------------------
# 5. GENERATE FINAL ANSWER
# -----------------------------

def generate_answer(user_query, docs):
    """
    Sends retrieved policy context to the LLM and generates a grounded answer.
    """

    context_text = "\n\n".join(
        [
            f"--- Context Chunk {i} ---\n{doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        ]
    )

    final_prompt = f"""
You are a customer service policy assistant for VelocityX.

Use only the policy context provided below to answer the user question.
If the answer is not available in the context, say:
"I could not find this in the provided VelocityX policy."

Be clear, practical, and policy-grounded.

Policy context:
{context_text}

User question:
{user_query}

Answer:
"""

    response = llm.invoke(final_prompt).content

    return response


# -----------------------------
# 6. FULL RAG PIPELINE
# -----------------------------

def run_rag_pipeline(user_query):
    """
    Complete Multi-Query RAG pipeline.
    """

    print("\n==================================================")
    print(f"USER QUERY: {user_query}")
    print("==================================================")

    docs = retrieve_with_multi_query(
        user_query=user_query,
        k=3
    )

    answer = generate_answer(
        user_query=user_query,
        docs=docs
    )

    return answer


# -----------------------------
# 7. TEST CASES
# -----------------------------

if __name__ == "__main__":

    test_cases = [
        "I ordered size 9 running shoes, but I received size 8. What resolution should I get?",
        "My shoes arrived with a damaged box and a torn upper. I reported it the next day. What evidence is required for a DOA claim?",
        "What does the VelocityX warranty not cover?",
        "What are the proof of purchase requirements for a VelocityX warranty claim?",
    ]

    for query in test_cases:
        answer = run_rag_pipeline(query)

        print("\n--- Final RAG Response ---")
        print(answer)
        print("\n\n")