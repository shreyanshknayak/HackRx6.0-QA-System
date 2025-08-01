import httpx      # An asynchronous HTTP client.
import os         # To handle file paths and create directories.
import asyncio    # To run synchronous libraries in an async environment.
from urllib.parse import unquote, urlparse # To get the filename from the URL.
import uuid       # To generate unique filenames if needed.
import fitz

from pydantic import HttpUrl
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import pymupdf4llm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import tempfile
import os
import argparse
from typing import Optional

# Ensure required libraries are installed.
# You can install them using:
# pip install llama_cloud_services pydantic python-dotenv

from llama_cloud_services import LlamaExtract, LlamaParse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Global variable for the extractor agent
llama_extract_agent = None
llama_parser = None


class Insurance(BaseModel):
    """
    A Pydantic model to define the data schema for extraction.
    The description helps guide the AI model.
    """
    headings: str = Field(description="An array of headings")


def initialize_llama_parser() -> LlamaParse:
    """
    Initializes and returns a configured LlamaParse client.
    This function should be called once at application startup.

    Returns:
        A LlamaParse instance ready for use.
    """
    global llama_parser
    if llama_parser is None:
        print("Initializing LlamaParse client")
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set.")
        
        print("Initializing LlamaParse client...")
        parser = LlamaParse(
            api_key=api_key,
            num_workers=4,
            verbose=True,
            language="en",
        )
        llama_parser = parser
        print("LlamaParse client initialized.")

def initialize_llama_extract_agent():
    global llama_extract_agent
    if llama_extract_agent is None:
        print("Initializing LlamaExtract client and getting agent...")
        try:
            extractor = LlamaExtract()
            llama_extract_agent = extractor.get_agent(name="insurance-parser")
            print("LlamaExtract agent initialized.")
        except Exception as e:
            print(f"Error initializing LlamaExtract agent: {e}")
            llama_extract_agent = None # Ensure it's None if there was an error


def extract_schema_from_file(file_path: str) -> Optional[Insurance]:
    if not os.path.exists(file_path):
        print(f"❌ Error: The file '{file_path}' was not found.")
        return None

    if llama_extract_agent is None:
        print("LlamaExtract agent not initialized. Attempting to initialize now.")
        initialize_llama_extract_agent()
        if llama_extract_agent is None:
            print("LlamaExtract agent failed to initialize. Cannot proceed with extraction.")
            return None

    print(f"🚀 Sending '{file_path}' to LlamaCloud for schema extraction...")

    try:
        result = llama_extract_agent.extract(file_path)

        if result and result.data:
            print("✅ Extraction successful!")
            return result.data
        else:
            print("⚠️ Extraction did not return any data.")
            return None

    except Exception as e:
        print(f"\n❌ An error occurred during the API call: {e}")
        print("Please check your API key, network connection, and file format.")
        return None



async def download_and_parse_document(doc_url: HttpUrl) -> str:
    """
    Asynchronously downloads a PDF document, saves it to a temporary file,
    and then parses it using the provided LlamaParse API client.

    Args:
        doc_url: The Pydantic-validated URL of the document to process.
        parser: An initialized LlamaParse client instance.

    Returns:
        A single string containing the document's content as structured Markdown.
    """
    print(f"Initiating download from: {doc_url}")

    temp_pdf_file_path = None

    if llama_parser is None:
        print("LlamaParse agent not initialized. Attempting to initialize now.")
        initialize_llama_parser()
        if llama_parser is None:
            print("LlamaParse agent failed to initialize. Cannot proceed with extraction.")
            return None
    try:
        # Step 1: Download the PDF file
        
        async with httpx.AsyncClient() as client:
            response = await client.get(str(doc_url), timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        
        doc_bytes = response.content
        print("Download successful.")

        # Step 2: Save the downloaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as temp_file:
            temp_file.write(doc_bytes)
            temp_pdf_file_path = temp_file.name
        
        print(f"Document saved temporarily at: {temp_pdf_file_path}")

        # Step 3: Parse the document using the provided LlamaParse client
        print("Parsing document with LlamaParse...")
        
        # LlamaParse's aparse method takes a list of file paths
        # and returns a list of LlamaParseResult objects.
        results = await llama_parser.aparse([temp_pdf_file_path])
        
        if not results or not results[0].get_markdown_documents():
            raise ValueError("LlamaParse did not return any content.")
        
        # Extract markdown from the result
        markdown_documents = results[0].get_markdown_documents(split_by_page=True)
        parsed_markdown = "\n\n".join([doc.text for doc in markdown_documents])
        
        print(f"Parsing complete. Extracted {len(parsed_markdown)} characters as Markdown.")
        
        return parsed_markdown

    except httpx.HTTPStatusError as e:
        print(f"Error downloading document: {e}")
        raise
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Step 4: Clean up the temporary file
        if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
            os.unlink(temp_pdf_file_path)
            print(f"Cleaned up temporary PDF file: {temp_pdf_file_path}")


