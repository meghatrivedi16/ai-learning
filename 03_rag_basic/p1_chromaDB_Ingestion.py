import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # current package, not langchain_community

# ------------------------------------------------------------
# Pipeline: Load PDF -> Split into chunks -> Embed -> Upsert into Chroma
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. Load environment variables
# ------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not found. Set it in your .env file before running.")

# ------------------------------------------------------------
# 2. Config
# ------------------------------------------------------------

PDF_PATH = os.getenv("PDF_PATH", "VelocityX Ecommerce Warranty, Returns & Exchange Policy.pdf")
COLLECTION_NAME = "velocity_x_policy"
PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ------------------------------------------------------------
# 3. PDF Loader
# ------------------------------------------------------------

if not os.path.isfile(PDF_PATH):
    sys.exit(f"ERROR: PDF not found at path: {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

if not documents:
    sys.exit(f"ERROR: No content extracted from {PDF_PATH}. "
              f"It may be empty, corrupted, or a scanned/image-only PDF requiring OCR.")

print(f"Total pages/documents loaded: {len(documents)}")

# ------------------------------------------------------------
# 4. PDF Splitter
# ------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = text_splitter.split_documents(documents)

if not chunks:
    sys.exit("ERROR: Text splitting produced 0 chunks. Check the source PDF content.")

print(f"Total chunks created: {len(chunks)}")

# ------------------------------------------------------------
# 5. Embeddings + Upsert into Chroma
# ------------------------------------------------------------

embeddings = OpenAIEmbeddings()

 
#EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims by default; cheaper + better than ada-002
#embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


try:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
except Exception as e:
    sys.exit(f"ERROR: Failed to embed/upsert into Chroma: {e}")

print("\n================ INGESTION STATUS ================")
print("Ingestion complete. Data stored in ChromaDB.")
print(f"Collection name: {COLLECTION_NAME}")
print(f"Persist directory: {PERSIST_DIR}")
print(f"Chunks stored: {len(chunks)}")