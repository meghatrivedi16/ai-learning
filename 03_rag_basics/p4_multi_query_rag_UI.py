"""
Streamlit UI for the Module 2 Multi-Query RAG app.
"""

from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

st.set_page_config(page_title="Multi-Query RAG", page_icon="🧠", layout="wide")


@st.cache_resource
def get_rag_components():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        collection_name="velocity_x_policy",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    return llm, vector_db


def get_multi_queries(llm, original_query):
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
    return [line.strip() for line in response.split("\n") if line.strip()]


def retrieve_with_multi_query(llm, vector_db, user_query, k=3):
    generated_queries = get_multi_queries(llm, user_query)
    all_queries = [user_query] + generated_queries

    all_retrieved_docs = []
    for query in all_queries:
        retrieved_docs = vector_db.similarity_search(query=query, k=k)
        all_retrieved_docs.extend(retrieved_docs)

    unique_docs = {}
    for doc in all_retrieved_docs:
        unique_docs[doc.page_content] = doc

    return generated_queries, list(unique_docs.values())


def generate_answer(llm, user_query, docs):
    context_text = "\n\n".join(
        [f"--- Context Chunk {i} ---\n{doc.page_content}" for i, doc in enumerate(docs, start=1)]
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

    return llm.invoke(final_prompt).content


def run_rag_pipeline(user_query, k=3):
    llm, vector_db = get_rag_components()
    generated_queries, docs = retrieve_with_multi_query(llm, vector_db, user_query, k=k)
    answer = generate_answer(llm, user_query, docs)
    return generated_queries, docs, answer


def main():
    if "query" not in st.session_state:
        st.session_state.query = ""

    st.title("Multi-Query RAG Assistant")
    st.caption("Ask a policy question and see the generated retrieval queries, matched context, and grounded answer.")

    with st.sidebar:
        st.header("Try an example")
        example_queries = [
            "I ordered size 9 running shoes, but I received size 8. What resolution should I get?",
            "My shoes arrived with a damaged box and a torn upper. I reported it the next day. What evidence is required for a DOA claim?",
            "What does the VelocityX warranty not cover?",
            "What are the proof of purchase requirements for a VelocityX warranty claim?",
        ]

        selected_example = st.selectbox("Choose a sample question", [""] + example_queries)
        if st.button("Load example") and selected_example:
            st.session_state.query = selected_example

        st.markdown("---")
        k = st.slider("Top context chunks per query", min_value=1, max_value=5, value=3)

    query = st.text_area("Ask a policy question", value=st.session_state.query, height=120)

    if st.button("Run RAG"):
        if not query.strip():
            st.warning("Please enter a question before running the pipeline.")
            return

        st.session_state.query = query

        with st.spinner("Generating alternate queries and crafting the answer..."):
            try:
                generated_queries, docs, answer = run_rag_pipeline(query, k=k)
            except Exception as exc:
                st.error(f"The RAG pipeline could not be completed: {exc}")
                return

        st.success("Answer generated")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Generated query variations")
        for item in generated_queries:
            st.write(f"- {item}")

        st.subheader("Retrieved context")
        if docs:
            for index, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "unknown")
                with st.expander(f"Context chunk {index} • Page {page}"):
                    st.write(doc.page_content)
        else:
            st.info("No relevant context was found for this question.")


if __name__ == "__main__":
    main()