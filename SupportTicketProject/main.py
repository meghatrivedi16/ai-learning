"""
main.py
=======
RAG+SQL support ticket system.

Mode - OpenAI only (via LangChain)
DB - Postgres

RUN THIS WITH:
    uv run uvicorn main:app --reload

Then open Postman and send a POST request as described below.
"""

import os
import re
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

app = FastAPI()

# Allows the browser-based frontend to call this API from a different origin
# (e.g. opening index.html directly, or a local dev server on another port).
# allow_origins=["*"] is fine for local development; a real deployment should
# list specific allowed domains instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ["DATABASE_URL"]

# One shared LLM client for routing + SQL generation + final answers.
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10)
sql_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=300)
answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=400)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

TICKETS_SCHEMA = """
tickets (
    ticket_id           SERIAL PRIMARY KEY,
    domain              TEXT,
    title               TEXT,
    description         TEXT,
    resolution_notes    TEXT,
    status              TEXT,   -- open / in_progress / resolved / closed
    priority            TEXT,   -- low / medium / high / critical
    category            TEXT,   -- refund / shipping / damaged_item / wrong_item / account / payment
    assignee            TEXT,
    reporter            TEXT,
    carrier             TEXT,
    tracking_number     TEXT,
    shipping_status     TEXT,
    estimated_delivery  DATE,
    order_id            TEXT,
    created_at          TIMESTAMP,
    resolved_at         TIMESTAMP
)
"""


class QuestionRequest(BaseModel):
    question: str


def classify_question(question: str) -> str:

    #
    # sample requests
    # {"question": "How many tickets are currently open?"}
    # {"question": "Find tickets similar to: customer says their package tracking hasn't updated in over a week"}
    # {"question": "Show me unresolved tickets similar to shipping delays"}
    #

    """
    Asks the LLM to decide whether this question needs a SQL query,
    a semantic (RAG) search, or both. Returns one of: "sql", "rag", "both".
    """
    router_prompt = f"""You are routing questions about a support tickets database.

The table has these structured fields: status, priority, category, assignee,
created_at, resolved_at, carrier, shipping_status.

The description field contains free-text ticket descriptions that can be
semantically searched.

Decide how to answer this question:
- "sql" if it only needs counting, filtering, or aggregating structured fields
- "rag" if it only needs finding tickets similar in meaning to some text
- "both" if it needs semantic matching AND structured filtering together
- "chitchat" if it's a greeting, small talk, or has nothing to do with support tickets at all

Respond with exactly one word: sql, rag, both, or chitchat.

Question: {question}"""

    response = router_llm.invoke(router_prompt)
    classification = response.content.strip().lower()

    if classification not in ("sql", "rag", "both", "chitchat"):
        classification = "both"

    return classification


def generate_chitchat_reply(question: str) -> str:
    """Handles greetings and off-topic messages with a short, friendly reply."""
    prompt = f"""You are a support ticket assistant. The user said something that
isn't a real question about tickets (a greeting, small talk, or off-topic).
Reply briefly and warmly, and mention you can help with questions about
support tickets (e.g. status, refunds, shipping, similar issues).

User said: {question}"""

    response = answer_llm.invoke(prompt)
    return response.content.strip()


def generate_sql(question: str) -> str:
    """
    Asks the LLM to write a SQL SELECT query against the known tickets schema.
    Returns the raw SQL text (no markdown fences, no explanation).
    """
    prompt = f"""Given this PostgreSQL table schema:

{TICKETS_SCHEMA}

Write a single SQL SELECT query that answers this question. Return ONLY the
raw SQL, no markdown code fences, no explanation, no semicolon at the end.

Question: {question}"""

    response = sql_llm.invoke(prompt)
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_select(sql: str) -> bool:
    """
    Basic safety check: only allow single SELECT statements.
    Blocks anything that could modify or destroy data.
    """
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return False
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", ";--", "grant"]
    return not any(word in normalized for word in forbidden)


def run_sql(sql: str) -> list[dict]:
    """Executes a validated SELECT query and returns rows as a list of dicts."""
    if not is_safe_select(sql):
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL generated: {sql}")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]


def embed_question(question: str) -> list[float]:
    """Embeds the question using OpenAI's embedding model. Correct method: embed_query (singular)."""
    return embeddings_model.embed_query(question)


def semantic_search(question: str, top_k: int = 5) -> list[dict]:
    """Finds the top_k tickets most similar in meaning to the question, using pgvector."""
    embedding = embed_question(question)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    query = """
        SELECT ticket_id, title, description, status, category,
               description_embedding <=> %s::vector AS distance
        FROM tickets
        WHERE description_embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT %s
    """

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (embedding_str, top_k))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]


def generate_final_answer(question: str, context: str) -> str:
    """Sends the gathered context to the LLM to produce a plain-English answer."""
    prompt = f"""Answer the user's question using ONLY the context below.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {question}"""

    response = answer_llm.invoke(prompt)
    return response.content.strip()


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is running at all."""
    return {"status": "running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    classification = classify_question(question)

    if classification == "chitchat":
        return {
            "you_asked": question,
            "classification": classification,
            "answer": generate_chitchat_reply(question),
        }

    context_parts = []

    if classification in ("sql", "both"):
        sql = generate_sql(question)
        sql_results = run_sql(sql)
        context_parts.append(f"SQL query used: {sql}\nSQL results: {sql_results}")

    if classification in ("rag", "both"):
        matches = semantic_search(question)
        context_parts.append(f"Semantically similar tickets: {matches}")

    context = "\n\n".join(context_parts)
    final_answer = generate_final_answer(question, context)

    return {
        "you_asked": question,
        "classification": classification,
        "answer": final_answer,
    }