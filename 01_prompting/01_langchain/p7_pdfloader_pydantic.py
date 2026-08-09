"""
Example: LangChain PDFLoader + LLM Processing + Pydantic Structured Output

Use case:
- Load a warranty and returns policy PDF
- Ask policy-related test questions
- Use GPT-4o-mini to answer in a structured Pydantic format

PDF used:
Warranty_Returns_Policy_LuxeThreads.pdf
"""

import os
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ------------------------------------------------------------
# 1. Load environment variables
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# 2. Define Pydantic structured output schema
# ------------------------------------------------------------

class PolicyAnswer(BaseModel):
    """
    Structured answer extracted from the policy document.
    """

    decision: Literal[
        "eligible",
        "not_eligible",
        "partially_eligible",
        "needs_more_information"
    ] = Field(
        description="Eligibility or decision based on the policy."
    )

    short_answer: str = Field(
        description="A concise customer-facing answer."
    )

    policy_basis: List[str] = Field(
        description="Specific policy points used to justify the answer."
    )

    required_customer_action: List[str] = Field(
        description="Actions the customer should take next."
    )

    expected_timeline: Optional[str] = Field(
        default=None,
        description="Relevant timeline, if mentioned in the policy."
    )

    contact_channel: Optional[str] = Field(
        default=None,
        description="Relevant email, phone, chat, or support channel."
    )


# ------------------------------------------------------------
# 3. Load PDF using LangChain PyPDFLoader
# ------------------------------------------------------------

PDF_PATH = "Warranty_Returns_Policy_LuxeThreads.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF.")

# Combine PDF page content into a single context string.
# For small PDFs, this is fine.
# For large PDFs, use chunking + retrieval instead.
policy_context = "\n\n".join(
    f"Page {i + 1}:\n{doc.page_content}"
    for i, doc in enumerate(documents)
)


# ------------------------------------------------------------
# 4. Create ChatOpenAI model
# ------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Convert the LLM into a structured-output model.
structured_llm = llm.with_structured_output(PolicyAnswer)


# ------------------------------------------------------------
# 5. Create prompt template
# ------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a policy analysis assistant.

Use only the policy text provided below to answer the customer question.
Do not invent rules that are not present in the policy.

Return your answer using the required structured schema.

Policy text:
{policy_context}
"""
    ),
    (
        "human",
        """
Customer question:
{customer_question}

Customer details:
{customer_details}
"""
    )
])


# ------------------------------------------------------------
# 6. Create LangChain chain
# ------------------------------------------------------------

chain = prompt | structured_llm


# ------------------------------------------------------------
# 7. Create 4 test cases
# ------------------------------------------------------------

test_cases = [
    {
        "name": "Test Case 1: Standard apparel return within 30 days",
        "customer_question": "Can I return a dress I bought?",
        "customer_details": """
Product: Dress
Category: Apparel
Delivery date: 20 days ago
Condition: Unworn, unused, original tags attached, original packaging available
Customer type: Standard customer
Proof of purchase: Available
"""
    },
    {
        "name": "Test Case 2: Fine jewelry return after 18 days by standard customer",
        "customer_question": "Can I return fine jewelry after 18 days?",
        "customer_details": """
Product: Fine jewelry necklace
Category: Fine Jewelry
Delivery date: 18 days ago
Condition: Unworn, original tags and packaging available
Customer type: Standard customer
Proof of purchase: Available
"""
    },
    {
        "name": "Test Case 3: Defective footwear received",
        "customer_question": "My shoes arrived with a sole separation defect. What can I do?",
        "customer_details": """
Product: Shoes
Category: Footwear
Delivery date: 10 days ago
Issue: Sole separation visible on arrival
Condition: Not used
Customer wants: Exchange or refund
Photos: Available
"""
    },
    {
        "name": "Test Case 4: International return for non-defective item",
        "customer_question": "I am an international customer. Who pays for return shipping?",
        "customer_details": """
Product: Handbag
Category: Accessories
Delivery date: 12 days ago
Issue: Customer changed mind
Condition: Unused, original packaging available
Customer location: Outside the US
Defective item: No
"""
    }
]


# ------------------------------------------------------------
# 8. Run test cases
# ------------------------------------------------------------

for test in test_cases:
    print("\n" + "=" * 80)
    print(test["name"])
    print("=" * 80)

    result: PolicyAnswer = chain.invoke({
        "policy_context": policy_context,
        "customer_question": test["customer_question"],
        "customer_details": test["customer_details"]
    })

    print("\nDecision:")
    print(result.decision)

    print("\nShort Answer:")
    print(result.short_answer)

    print("\nPolicy Basis:")
    for item in result.policy_basis:
        print(f"- {item}")

    print("\nRequired Customer Action:")
    for action in result.required_customer_action:
        print(f"- {action}")

    print("\nExpected Timeline:")
    print(result.expected_timeline)

    print("\nContact Channel:")
    print(result.contact_channel)