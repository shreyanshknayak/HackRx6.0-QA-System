import os
import json
import tempfile
import requests
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Union, Any, Optional
from dotenv import load_dotenv
import asyncio
import httpx
import time
from urllib.parse import urlparse, unquote
import uuid
import re

# Import LangChain Document and text splitter
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from processing_utility import download_and_parse_document, extract_schema_from_file, initialize_llama_extract_agent


# Import functions and constants from the colbert_utils.py file
# Make sure colbert_utils.py is in the same directory or accessible via PYTHONPATH
from rag_utils import (
    process_markdown_with_manual_sections,
    perform_hybrid_search,
    generate_answer_with_groq,
    initialize_hybrid_search_models,
    load_embedding_model_at_startup,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    GROQ_MODEL_NAME
)

load_dotenv()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRX RAG API",
    description="API for Retrieval-Augmented Generation from PDF documents.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    load_embedding_model_at_startup() # From rag_utils
    initialize_llama_extract_agent() # From processing_utility


# --- Groq API Key Setup ---
# It's highly recommended to set this as an environment variable in production.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "NOT_FOUND")
if GROQ_API_KEY == "NOT_FOUND":
    print("WARNING: GROQ_API_KEY is using a placeholder or hardcoded value. Please set GROQ_API_KEY environment variable for production.")

# --- Authorization Token Setup ---
# Set your authorization token as an environment variable.
# For example, in your .env file: AUTHORIZATION_TOKEN="your_secret_token_here"
EXPECTED_AUTH_TOKEN = os.getenv("AUTHORIZATION_TOKEN")
if not EXPECTED_AUTH_TOKEN:
    print("WARNING: AUTHORIZATION_TOKEN environment variable is not set. Authorization will not work as expected.")


# --- Pydantic Models for Request and Response ---
class RunRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class Answer(BaseModel):
    answer: str

class RunResponse(BaseModel):
    answers: List[Answer]
    processing_time: float # Added to include the total processing time

# --- Security Dependency ---
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Bearer token in the Authorization header.
    """
    if not EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authorization token not configured on the server."
        )
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Pseudo-functions (Replace with actual implementations if needed) ---

async def process_single_question(question: str, processed_documents: List[Dict[str, Any]], groq_api_key: str) -> Answer:
    """
    Processes a single question: performs vector search and generates an answer.
    This function is designed to be run concurrently.
    """
    print(f"Starting processing for question: '{question}'")
    # Perform vector search
    retrieved_results = await perform_hybrid_search(question, TOP_K_CHUNKS)

    if retrieved_results:
        # Generate answer using Groq
        answer_text = await generate_answer_with_groq(question, retrieved_results, GROQ_API_KEY)
    else:
        answer_text = "No relevant information found in the document to answer this question."

    print(f"Finished processing for question: '{question}'")
    return Answer(answer=answer_text)

@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_pipeline(
    request: RunRequest,
    authorized: bool = Depends(verify_token) # Add this line to enforce authorization
):
    """
    Runs the RAG pipeline for a given PDF document (converted to Markdown internally)
    and a list of questions, parallelizing the processing of each question.
    Uses a hybrid search method (BM25 + Dense Vectors).
    Requires a valid Bearer token in the Authorization header.
    """
    pdf_url = request.documents
    questions = request.questions

    local_markdown_path = None # Path to the temporary markdown file

    try:
        # Step 1: Download PDF and parse to Markdown
        # This function should return the path to the converted markdown file
        markdown_content = await download_and_parse_document(pdf_url)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.md') as temp_md_file:
            temp_md_file.write(markdown_content)
            local_markdown_path = temp_md_file.name

        # Step 2: Extract headings JSON from the markdown file
        headings_json = extract_schema_from_file(local_markdown_path)
        
        # Optional: For debugging, write headings to output.json
        with open("headings.json", 'w', encoding='utf-8') as f:
            json.dump(headings_json, f, indent=4, ensure_ascii=False)

        if not headings_json or not headings_json.get("headings"):
            raise HTTPException(status_code=400, detail="Could not retrieve valid headings from the provided document.")

        # Step 3: Process Markdown with manual sections to get chunks with metadata
        print("Processing Markdown into chunks with manual sections...")
        processed_documents = process_markdown_with_manual_sections(
            local_markdown_path,
            headings_json,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )
        if not processed_documents:
            raise HTTPException(status_code=500, detail="Failed to process document into chunks.")

        # Step 4: Initialize Hybrid Search Models (BM25 and Sentence Transformers)
        # This must be done ONCE after processing all documents.
        print("Initializing Hybrid Search models...")
        initialize_hybrid_search_models(processed_documents)
        print("Hybrid Search models initialized.")

        # Step 5: Parallelize question processing
        print(f"Processing {len(questions)} questions in parallel...")
        start_time = time.perf_counter() # Start timing
        tasks = [
            process_single_question(question, processed_documents, GROQ_API_KEY)
            for question in questions
        ]
        all_answers = await asyncio.gather(*tasks)
        end_time = time.perf_counter() # End timing
        total_processing_time = end_time - start_time
        print("All questions processed.")

        return RunResponse(answers=all_answers, processing_time=total_processing_time)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # Clean up the temporary markdown file
        if local_markdown_path and os.path.exists(local_markdown_path):
            os.unlink(local_markdown_path)
            print(f"Cleaned up temporary markdown file: {local_markdown_path}")
