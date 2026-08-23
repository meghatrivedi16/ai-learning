"""
RAPTOR RAG - Warranty & Returns Policy Implementation
======================================================
Hierarchical RAG using LangChain + ChromaDB

Usage:
    1. Set your OpenAI API key in environment or .env file
    2. First run (build indexes): python raptor_rag_warranty.py --build
    3. Subsequent runs (use existing indexes): python raptor_rag_warranty.py

Author: Agentic AI Architecture Course
Document: Luxe Threads Warranty & Returns Policy
"""

import os
import argparse
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Paths for persistent storage
CHROMA_STANDARD_PATH = "./chroma_standard_warranty"
CHROMA_RAPTOR_PATH = "./chroma_raptor_warranty"

# Path to the warranty PDF (in same folder as script or uploads)
PDF_PATH = "Warranty_Returns_Policy_LuxeThreads.pdf"
UPLOADS_PATH = "Warranty_Returns_Policy_LuxeThreads.pdf"

# ============================================================
# IMPORTS (Updated for LangChain 0.2+ / 0.3+)
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture

# ============================================================
# STEP 1: LOAD PDF FROM UPLOADS OR LOCAL
# ============================================================
def get_pdf_path():
    """Get the PDF path - check uploads folder first, then local"""
    if os.path.exists(UPLOADS_PATH):
        print(f"✅ Found PDF in uploads: {UPLOADS_PATH}")
        return UPLOADS_PATH
    elif os.path.exists(PDF_PATH):
        print(f"✅ Found PDF locally: {PDF_PATH}")
        return PDF_PATH
    else:
        raise FileNotFoundError(
            f"PDF not found. Please ensure '{PDF_PATH}' exists in the current directory "
            f"or uploads folder."
        )

# ============================================================
# STEP 2: LOAD AND CHUNK DOCUMENT (Level 0 - Leaf Nodes)
# ============================================================
def load_and_chunk(pdf_path: str, chunk_size: int = 500) -> List[Document]:
    """Load PDF and split into chunks"""
    print("\n📖 Loading and chunking document...")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    
    # Add metadata for Level 0
    for i, chunk in enumerate(chunks):
        chunk.metadata["level"] = 0
        chunk.metadata["node_type"] = "chunk"
        chunk.metadata["node_id"] = f"L0_{i}"
    
    print(f"   Created {len(chunks)} chunks (Level 0)")
    return chunks

# ============================================================
# STEP 3: CLUSTER SIMILAR CHUNKS
# ============================================================
def cluster_chunks(chunks: List[Document], embeddings, n_clusters: int) -> Dict[int, List[Document]]:
    """Group similar chunks using GMM clustering"""
    print(f"\n🔗 Clustering {len(chunks)} chunks into {n_clusters} groups...")
    
    # Get embeddings
    texts = [c.page_content for c in chunks]
    vectors = np.array(embeddings.embed_documents(texts))
    
    # Cluster
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(vectors)  # look like [0, 1, 0, 2, ...]
    
    # Group by cluster
    clusters = {}
    for chunk, label in zip(chunks, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunk)
    
    return clusters

# ============================================================
# STEP 4: SUMMARIZE CLUSTERS (Level 1+ - Summary Nodes)
# ============================================================
def summarize_clusters(clusters: Dict[int, List[Document]], llm, level: int) -> List[Document]:
    """Generate summaries for each cluster"""
    print(f"\n📝 Creating Level {level} summaries...")
    
    prompt = ChatPromptTemplate.from_template("""
Summarize these related document sections. Preserve key facts, numbers, and concepts:

{text}

Concise Summary:""")
    
    # Use LCEL chain style
    chain = prompt | llm | StrOutputParser()
    summaries = []
    
    for cluster_id, docs in clusters.items():
        # Combine cluster texts
        combined = "\n\n---\n\n".join([d.page_content for d in docs])
        if len(combined) > 8000:
            combined = combined[:8000]
        
        # Generate summary
        summary_text = chain.invoke({"text": combined})
        
        summary_doc = Document(
            page_content=summary_text,
            metadata={
                "level": level,
                "node_type": "summary",
                "node_id": f"L{level}_{cluster_id}",
                "children": len(docs)
            }
        )
        summaries.append(summary_doc)
        print(f"   Cluster {cluster_id}: {len(docs)} chunks → 1 summary")
    
    print(f"   Created {len(summaries)} summaries (Level {level})")
    return summaries

# ============================================================
# STEP 5: BUILD RAPTOR TREE
# ============================================================
def build_raptor_tree(chunks: List[Document], embeddings, llm, max_levels: int = 2) -> List[Document]:
    """Build hierarchical RAPTOR tree"""
    print("\n" + "="*50)
    print("🌲 BUILDING RAPTOR TREE")
    print("="*50)
    
    all_nodes = list(chunks)  # Start with Level 0
    current_docs = chunks
    
    for level in range(1, max_levels + 1):
        if len(current_docs) < 6:
            print(f"\n   Stopping: too few docs for Level {level}")
            break
        
        # Determine clusters (reduce by ~75% each level)
        n_clusters = max(3, len(current_docs) // 4)
        
        # Cluster and summarize
        clusters = cluster_chunks(current_docs, embeddings, n_clusters)
        summaries = summarize_clusters(clusters, llm, level)
        
        all_nodes.extend(summaries)
        current_docs = summaries
    
    # Print tree stats
    print("\n" + "="*50)
    print("📊 RAPTOR TREE COMPLETE")
    level_counts = {}
    for doc in all_nodes:
        lvl = doc.metadata.get("level", 0)
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    
    for lvl in sorted(level_counts.keys()):
        node_type = "chunks" if lvl == 0 else "summaries"
        print(f"   Level {lvl}: {level_counts[lvl]} {node_type}")
    print(f"   Total: {len(all_nodes)} nodes")
    print("="*50)
    
    return all_nodes

# ============================================================
# STEP 6: CREATE VECTOR STORES
# ============================================================
def create_standard_index(chunks: List[Document], embeddings) -> Chroma:
    """Create standard RAG index (chunks only)"""
    print("\n🔷 Creating Standard RAG index...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="standard_rag_warranty",
        persist_directory=CHROMA_STANDARD_PATH
    )

def create_raptor_index(all_nodes: List[Document], embeddings) -> Chroma:
    """Create RAPTOR index (all levels)"""
    print("\n🌲 Creating RAPTOR index...")
    return Chroma.from_documents(
        documents=all_nodes,
        embedding=embeddings,
        collection_name="raptor_rag_warranty",
        persist_directory=CHROMA_RAPTOR_PATH
    )

def load_existing_indexes(embeddings) -> tuple:
    """Load existing vector stores from disk"""
    print("\n📂 Loading existing vector stores...")
    
    standard_db = Chroma(
        collection_name="standard_rag_warranty",
        embedding_function=embeddings,
        persist_directory=CHROMA_STANDARD_PATH
    )
    
    raptor_db = Chroma(
        collection_name="raptor_rag_warranty",
        embedding_function=embeddings,
        persist_directory=CHROMA_RAPTOR_PATH
    )
    
    print(f"   ✅ Loaded Standard RAG index from {CHROMA_STANDARD_PATH}")
    print(f"   ✅ Loaded RAPTOR index from {CHROMA_RAPTOR_PATH}")
    
    return standard_db, raptor_db

def indexes_exist() -> bool:
    """Check if vector store indexes already exist"""
    return os.path.exists(CHROMA_STANDARD_PATH) and os.path.exists(CHROMA_RAPTOR_PATH)

# ============================================================
# STEP 7: QUERY FUNCTIONS
# ============================================================
def query_rag(question: str, vectorstore: Chroma, llm, k: int = 5) -> Dict:
    """Query a RAG system"""
    # Retrieve
    docs = vectorstore.similarity_search(question, k=k)
    
    # Build context
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    
    # Generate answer
    prompt = ChatPromptTemplate.from_template("""
Answer based on the context provided. Be concise but complete.

Context:
{context}

Question: {question}

Answer:""")
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    # Get level distribution (for RAPTOR)
    levels = {}
    for d in docs:
        lvl = d.metadata.get("level", "?")
        levels[lvl] = levels.get(lvl, 0) + 1
    
    return {"answer": answer, "levels": levels}

# ============================================================
# STEP 8: COMPARE STANDARD RAG VS RAPTOR
# ============================================================
def compare(question: str, standard_db: Chroma, raptor_db: Chroma, llm):
    """Compare Standard RAG vs RAPTOR on a question"""
    print("\n" + "="*60)
    print(f"❓ {question}")
    print("="*60)
    
    # Standard RAG
    print("\n🔷 STANDARD RAG:")
    print("-"*40)
    std_result = query_rag(question, standard_db, llm)
    print(std_result["answer"])
    
    # RAPTOR RAG
    print("\n🌲 RAPTOR RAG:")
    print("-"*40)
    rap_result = query_rag(question, raptor_db, llm)
    print(rap_result["answer"])
    print(f"\n[Retrieved from levels: {rap_result['levels']}]")

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RAPTOR RAG for Luxe Threads Warranty & Returns Policy"
    )
    parser.add_argument(
        "--build", 
        action="store_true",
        help="Build/rebuild the vector stores (required on first run)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if indexes exist"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("🌲 RAPTOR RAG - Luxe Threads Warranty & Returns Policy")
    print("="*60)
    
    # Check API key
    if not OPENAI_API_KEY:
        print("\n❌ Please set your OPENAI_API_KEY environment variable!")
        return
    
    # Initialize models
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Determine whether to build or load indexes
    should_build = args.build or args.force_rebuild or not indexes_exist()
    
    if should_build:
        if indexes_exist() and not args.force_rebuild and not args.build:
            print("\n⚠️  No existing indexes found. Building new indexes...")
        elif args.force_rebuild:
            print("\n🔄 Force rebuilding indexes...")
        else:
            print("\n🔨 Building new indexes...")
        
        # Get PDF path and process document
        pdf_path = get_pdf_path()
        chunks = load_and_chunk(pdf_path)
        
        # Build RAPTOR tree
        raptor_nodes = build_raptor_tree(chunks, embeddings, llm, max_levels=2)
        
        # Create indexes
        standard_db = create_standard_index(chunks, embeddings)
        raptor_db = create_raptor_index(raptor_nodes, embeddings)
        
        print("\n✅ Both indexes built and persisted!")
    else:
        print("\n📂 Using existing indexes (use --build to rebuild)")
        standard_db, raptor_db = load_existing_indexes(embeddings)
    
    # ========================================================
    # TEST COMPARISONS - Warranty & Returns Policy Questions
    # ========================================================
    print("\n" + "="*60)
    print("📊 COMPARISON TESTS")
    print("="*60)
    
    # Test 1: Specific factual question (both should do well)
    compare(
        "What is the return window for VIP members purchasing fine jewelry?",
        standard_db, raptor_db, llm
    )
    
    # Test 2: Abstract/policy philosophy question (RAPTOR should excel)
    compare(
        "What is Luxe Threads' overall approach to customer satisfaction and returns?",
        standard_db, raptor_db, llm
    )
    
    # Test 3: Cross-section question spanning multiple policy areas
    compare(
        "How does Luxe Threads handle situations where a customer receives a damaged or defective item?",
        standard_db, raptor_db, llm
    )
    
    # Test 4: Warranty specifics
    compare(
        "What warranty coverage does Luxe Threads provide for leather goods and what defects are covered?",
        standard_db, raptor_db, llm
    )
    
    # Test 5: Refund process question
    compare(
        "What are all the conditions and requirements a customer must meet to successfully return, exchange, or claim warranty on an item?",
        standard_db, raptor_db, llm
    )
    
    # Test 6: Exclusions and non-returnable items
    compare(
        "What are all the different time windows and deadlines a customer needs to be aware of when dealing with Luxe Threads?",
        standard_db, raptor_db, llm
    )
    
    # ========================================================
    # INTERACTIVE MODE
    # ========================================================
    print("\n" + "="*60)
    print("🔍 INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q', '']:
            break
        compare(question, standard_db, raptor_db, llm)
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()