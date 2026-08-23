"""
embed_tickets.py
=================
One-time script: reads every ticket's description from Postgres, generates
an embedding for it using OpenAI, and stores that embedding in the
description_embedding column (added earlier via ALTER TABLE).

Run this once now to populate all existing tickets. Re-run it later if you
add new tickets that don't have embeddings yet (it only processes rows
where description_embedding IS NULL, so it's safe to re-run anytime).

USAGE:
    uv run python embed_tickets.py
"""

import os
import psycopg
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY, OPENAI_API_KEY, DATABASE_URL from .env

DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, matches the column we created
BATCH_SIZE = 20  # how many ticket descriptions to send to OpenAI per API call

client = OpenAI()  # reads OPENAI_API_KEY from environment automatically


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Send a batch of texts to OpenAI and get back a list of embedding vectors."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data is in the same order as the input texts
    return [item.embedding for item in response.data]


def vector_to_pg_literal(vector: list[float]) -> str:
    """Convert a Python list of floats into the string format pgvector expects, e.g. '[0.1,0.2,0.3]'."""
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Only fetch tickets that don't have an embedding yet
            cur.execute(
                "SELECT ticket_id, description FROM tickets WHERE description_embedding IS NULL ORDER BY ticket_id"
            )
            rows = cur.fetchall()

            if not rows:
                print("No tickets need embedding. Everything is already up to date.")
                return

            print(f"Found {len(rows)} tickets without embeddings. Generating...")

            processed = 0
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                ticket_ids = [row[0] for row in batch]
                descriptions = [row[1] for row in batch]

                embeddings = get_embeddings(descriptions)

                # Write each embedding back to its corresponding row
                for ticket_id, embedding in zip(ticket_ids, embeddings):
                    pg_vector_str = vector_to_pg_literal(embedding)
                    cur.execute(
                        "UPDATE tickets SET description_embedding = %s WHERE ticket_id = %s",
                        (pg_vector_str, ticket_id),
                    )

                conn.commit()  # save progress after each batch
                processed += len(batch)
                print(f"  Processed {processed}/{len(rows)} tickets...")

            print(f"Done. {processed} tickets now have embeddings.")


if __name__ == "__main__":
    main()