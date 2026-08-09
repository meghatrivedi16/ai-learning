import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(system, user, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content

# Step 1: generate ideas
ideas = ask(
    system="You generate blog post ideas.",
    user="Give me 3 blog post title ideas about RAG (Retrieval-Augmented Generation) for beginners."
)
print("STEP 1 - Ideas:\n", ideas, "\n")

# Step 2: outline the best one
outline = ask(
    system="You create structured blog post outlines.",
    user=f"From these ideas, pick the best one for a beginner audience and create a 5-point outline:\n\n{ideas}"
)
print("STEP 2 - Outline:\n", outline, "\n")

# Step 3: write the post from the outline
post = ask(
    system="You are a technical writer who explains AI concepts clearly for beginners.",
    user=f"Write a 400-word blog post following this outline:\n\n{outline}",
    temperature=0.6
)
print("STEP 3 - Final Post:\n", post)