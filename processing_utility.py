import httpx      # An asynchronous HTTP client.
import os         # To handle file paths and create directories.
import asyncio    # To run synchronous libraries in an async environment.
from urllib.parse import unquote, urlparse # To get the filename from the URL.
import uuid       # To generate unique filenames if needed.

from pydantic import HttpUrl
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


import os
import argparse
from typing import Optional

# Ensure required libraries are installed.
# You can install them using:
# pip install llama_cloud_services pydantic python-dotenv

from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Global variable for the extractor agent
llama_extract_agent = None


class Insurance(BaseModel):
    """
    A Pydantic model to define the data schema for extraction.
    The description helps guide the AI model.
    """
    headings: str = Field(description="An array of headings")


class Insurance(BaseModel):
    headings: str = Field(description="An array of headings")

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
    Asynchronously downloads a document, saves it to a local directory,
    and then parses it using LangChain's PyMuPDF4LLMLoader.

    Args:
        doc_url: The Pydantic-validated URL of the document to process.

    Returns:
        A single string containing the document's content as structured Markdown.
    """
    print(f"Initiating download from: {doc_url}")
    try:
        # Create the local storage directory if it doesn't exist.
        LOCAL_STORAGE_DIR = "data/"
        os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

        async with httpx.AsyncClient() as client:
            response = await client.get(str(doc_url), timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        
        doc_bytes = response.content
        print("Download successful.")

        # --- Logic to determine the local filename ---
        # Parse the URL to extract the path.
        parsed_path = urlparse(str(doc_url)).path
        # Get the last part of the path and decode URL-encoded characters (like %20 for space).
        filename = unquote(os.path.basename(parsed_path))
        
        # If the filename is empty, create a unique one.
        if not filename:
            filename = f"{uuid.uuid4()}.pdf"
            
        # Construct the full path where the file will be saved.
        local_file_path = os.path.join(LOCAL_STORAGE_DIR, filename)

        # Save the downloaded document to the local file.
        with open(local_file_path, "wb") as f:
            f.write(doc_bytes)
        
        print(f"Document saved locally at: {local_file_path}")
        print("Parsing document with LangChain's PyMuPDF4LLMLoader...")

        # The loader's 'load' method is synchronous. Run it in a separate thread.
        def load_document():
            loader = PyMuPDF4LLMLoader(local_file_path)
            documents = loader.load()
            return documents

        documents = await asyncio.to_thread(load_document)
        
        if documents:
            parsed_markdown = "\n\n".join([doc.page_content for doc in documents])
            print(f"Parsing complete. Extracted {len(parsed_markdown)} characters as Markdown.")
            # The local file is NOT deleted, as requested.
            '''with open("sample_schema.json", 'r') as file:
                # Load the JSON data from the file into a Python variable (dictionary or list)
                data_variable = json.load(file)'''

            #await process_markdown_with_manual_sections(parsed_markdown, data_variable, chunk_size = 1000, chunk_overlap =200)
            print(f"Markdown successfully saved to {filename}")
            return parsed_markdown
            return filename
        else:
            raise ValueError("PyMuPDF4LLMLoader did not return any content.")

    except httpx.HTTPStatusError as e:
        print(f"Error downloading document: {e}")
        raise
    except Exception as e:
        print(f"Error during processing: {e}")
        raise