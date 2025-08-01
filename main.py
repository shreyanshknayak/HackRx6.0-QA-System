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

#Import LangChain Document and text splitter
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming these are available in your environment
from processing_utility import download_and_parse_document, extract_schema_from_file, initialize_llama_extract_agent, initialize_llama_parser
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
    initialize_llama_parser() # From processing_utility


# --- Groq API Key Setup ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "NOT_FOUND")
if GROQ_API_KEY == "NOT_FOUND":
    print("WARNING: GROQ_API_KEY is using a placeholder or hardcoded value. Please set GROQ_API_KEY environment variable for production.")

# --- Authorization Token Setup ---
# Set your authorization token as an environment variable.
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
    #processing_time: float
    #step_timings: dict # New field for detailed timings

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


@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_pipeline(
    request: RunRequest,
    authorized: bool = Depends(verify_token)
):
    """
    Runs the RAG pipeline for a given PDF document (converted to Markdown internally)
    and a list of questions.
    """
    pdf_url = request.documents
    questions = request.questions
    local_markdown_path = None
    step_timings = {}

    start_time_total = time.perf_counter()

    try:
        # 1. Parsing: Download PDF and parse to Markdown
        start_time = time.perf_counter()
        markdown_content = await download_and_parse_document(pdf_url)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.md') as temp_md_file:
            temp_md_file.write(markdown_content)
            local_markdown_path = temp_md_file.name
        end_time = time.perf_counter()
        step_timings["parsing_to_markdown"] = end_time - start_time
        print(f"Parsing to Markdown took {step_timings['parsing_to_markdown']:.2f} seconds.")

        # 2. Headings Generation: Extract headings JSON
        start_time = time.perf_counter()
        headings_json = extract_schema_from_file(local_markdown_path)
        if not headings_json or not headings_json.get("headings"):
            raise HTTPException(status_code=400, detail="Could not retrieve valid headings from the provided document.")
        end_time = time.perf_counter()
        step_timings["headings_generation"] = end_time - start_time
        print(f"Headings Generation took {step_timings['headings_generation']:.2f} seconds.")

        # 3. Chunk Generation: Process Markdown into chunks
        start_time = time.perf_counter()
        processed_documents = process_markdown_with_manual_sections(
            local_markdown_path,
            headings_json,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )
        if not processed_documents:
            raise HTTPException(status_code=500, detail="Failed to process document into chunks.")
        end_time = time.perf_counter()
        step_timings["chunk_generation"] = end_time - start_time
        print(f"Chunk Generation took {step_timings['chunk_generation']:.2f} seconds.")
        
        # 4. Model Initialization and Embeddings Pre-computation
        start_time = time.perf_counter()
        document_embeddings = initialize_hybrid_search_models(processed_documents)
        end_time = time.perf_counter()
        step_timings["model_initialization"] = end_time - start_time
        print(f"Model initialization took {step_timings['model_initialization']:.2f} seconds.")

        # 5. Concurrent Query Processing (Search and Generation)
        start_time_query_processing = time.perf_counter()

        # Search Phase
        batch_size = 3
        all_retrieved_results = []
        print(f"Starting concurrent search in batches of {batch_size}...")
        
        for i in range(0, len(questions), batch_size):
            current_batch_questions = questions[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} with {len(current_batch_questions)} queries.")
            
            search_tasks = [
                asyncio.to_thread(perform_hybrid_search, question, TOP_K_CHUNKS, document_embeddings)
                for question in current_batch_questions
            ]
            batch_results = await asyncio.gather(*search_tasks)
            all_retrieved_results.extend(batch_results)
            
        print("Search phase completed for all queries.")
        
        # Generation Phase
        print(f"Starting concurrent answer generation for {len(questions)} questions...")
        generation_tasks = []
        for question, retrieved_results in zip(questions, all_retrieved_results):
            if retrieved_results:
                generation_tasks.append(
                    generate_answer_with_groq(question, retrieved_results, GROQ_API_KEY)
                )
            else:
                no_info_future = asyncio.Future()
                no_info_future.set_result("No relevant information found in the document to answer this question.")
                generation_tasks.append(no_info_future)
        
        all_answer_texts = await asyncio.gather(*generation_tasks)
        
        end_time_query_processing = time.perf_counter()
        step_timings["query_processing"] = end_time_query_processing - start_time_query_processing
        print(f"Total query processing took {step_timings['query_processing']:.2f} seconds.")
        
        end_time_total = time.perf_counter()
        total_processing_time = end_time_total - start_time_total
        print("All questions processed.")
        
        all_answers = [Answer(answer=answer_text) for answer_text in all_answer_texts]

        return RunResponse(
            answers=all_answers
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        if local_markdown_path and os.path.exists(local_markdown_path):
            os.unlink(local_markdown_path)
            print(f"Cleaned up temporary markdown file: {local_markdown_path}")
