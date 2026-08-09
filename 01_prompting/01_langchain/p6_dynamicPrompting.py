# dynamic_prompting_langchain.py

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()

# 1. Initialize model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4
)

# 2. Dynamic prompt builder
def build_dynamic_prompt(
    role: str,
    task: str,
    audience: str,
    tone: str,
    output_format: str,
    context: str | None = None,
    constraints: str | None = None
):
    """
    Builds a dynamic prompt based on runtime inputs.
    """

    system_prompt = """
You are acting as a {role}.

Your goal is to complete the user task clearly and accurately.

Audience:
{audience}

Tone:
{tone}

Output format:
{output_format}
"""

    if constraints:
        system_prompt += """

Important constraints:
{constraints}
"""

    if context:
        human_prompt = """
Use the following context while answering:

Context:
{context}

Task:
{task}
"""
    else:
        human_prompt = """
Task:
{task}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    return prompt


# 3. Runtime input values
input_data = {
    "role": "AI product strategy consultant",
    "task": "Create a short product concept for an AI voice coach for leadership development.",
    "audience": "startup founders and product leaders",
    "tone": "professional, practical, and easy to understand",
    "output_format": """"structured bullets with clear sections, use emojis, return a heavily formatted markdown. Create three sections -
     "Product Concept", "Key Features", "Unique Value Proposition" . Use detailed explanations in each section, each feature.
     Return a json format with three keys - "product_concept", "key_features", "unique_value_proposition" and the value of each key should be the content of the respective section in markdown format."
     """, 
    "context": """
The product is mobile-first with a desktop view.
It uses a voice AI agent to understand the learner's current leadership state.
It provides knowledge snippets, assessments, coaching nudges, workplace guidance,
career growth support, skill growth support, and embedded feedback.
""",
    "constraints": "Keep the answer concise. Avoid technical jargon."
}


# 4. Build prompt dynamically
prompt = build_dynamic_prompt(
    role=input_data["role"],
    task=input_data["task"],
    audience=input_data["audience"],
    tone=input_data["tone"],
    output_format=input_data["output_format"],
    context=input_data.get("context"),
    constraints=input_data.get("constraints")
)


# 5. Create chain using LCEL
chain = prompt | llm


# 6. Invoke the chain
response = chain.invoke(input_data)

print(response.content)