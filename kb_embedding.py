import os
import json
import time
import glob
import numpy as np
import faiss
import tiktoken
from pypdf import PdfReader
from openai import OpenAI

# ==========================================
# 1. Configuration
# ==========================================
# IMPORTANT: Never hardcode your API key. Always use environment variables.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS_PER_CHUNK = 170  # 500 tokens is a great sweet spot for RAG retrieval
BATCH_SIZE = 100            # How many chunks to send to OpenAI at once

# ==========================================
# 2. Text Extraction & Chunking
# ==========================================
def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def chunk_text_by_tokens(text):
    """Splits text into paragraphs, then enforces a strict token limit."""
    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    
    # First, split the text naturally by paragraphs
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    final_chunks = []
    
    for para in raw_paragraphs:
        tokens = encoder.encode(para)
        
        # If the paragraph is within our limit, keep it intact
        if len(tokens) <= MAX_TOKENS_PER_CHUNK:
            final_chunks.append(para)
        else:
            # If it's too big, strictly slice it by exact token counts
            for i in range(0, len(tokens), MAX_TOKENS_PER_CHUNK):
                chunk_tokens = tokens[i : i + MAX_TOKENS_PER_CHUNK]
                chunk_text = encoder.decode(chunk_tokens)
                final_chunks.append(chunk_text)
                
    return final_chunks

# ==========================================
# 3. Safe API Execution
# ==========================================
def get_embeddings_safely(documents):
    """Fetches embeddings in smaller batches to avoid OpenAI rate limits."""
    print(f"Embedding {len(documents)} chunks in batches of {BATCH_SIZE}...")
    all_embeddings = []
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"Processing batch {(i // BATCH_SIZE) + 1} / {(len(documents) // BATCH_SIZE) + 1}...")
        
        try:
            response = openai_client.embeddings.create(
                input=batch, 
                model=EMBEDDING_MODEL
            )
            all_embeddings.extend([item.embedding for item in response.data])
            time.sleep(1) # Tiny pause to respect rate limits
            
        except Exception as e:
            print(f"API Error on batch {(i // BATCH_SIZE) + 1}: {e}")
            print("Waiting 10 seconds before retrying...")
            time.sleep(10)
            response = openai_client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            all_embeddings.extend([item.embedding for item in response.data])
            
    return all_embeddings

# ==========================================
# 4. Main Pipeline
# ==========================================
def build_knowledge_base_from_directory(directory_path="dataset/kb/pdfs/"):
    print(f"Starting ingestion pipeline for PDFs in {directory_path}...")
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} not found.")
        return

    # 1. Find all PDFs in the directory
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory_path}.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s). Extracting text...")

    # 2. Extract and chunk text from all PDFs
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Processing: {os.path.basename(pdf_file)}")
        raw_text = extract_text_from_pdf(pdf_file)
        
        if raw_text.strip():
            chunks = chunk_text_by_tokens(raw_text)
            all_documents.extend(chunks)
        else:
            print(f"Warning: No usable text extracted from {pdf_file}.")

    if not all_documents:
        print("No usable text found across all PDFs after cleaning.")
        return

    print(f"Text successfully split into {len(all_documents)} total token-optimized chunks.")

    # 3. Embed (with rate-limit protection)
    embeddings = get_embeddings_safely(all_documents)

    # 4. Build and Save FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save the index and the reference documents
    faiss.write_index(index, "multi_pdf_faiss_index.bin")
    with open("multi_pdf_documents.json", 'w', encoding='utf-8') as f:
        json.dump(all_documents, f)
        
    print("Success! 'multi_pdf_faiss_index.bin' and 'multi_pdf_documents.json' are ready for your Flask app.")

if __name__ == "__main__":
    # Point this to the folder containing your PDFs
    build_knowledge_base_from_directory("kb/")