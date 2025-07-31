import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import AsyncGroq
import json
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- Configuration (can be overridden by the calling app) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5
GROQ_MODEL_NAME = "llama3-8b-8192"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good general-purpose embedding model

# --- Global instances for models (to avoid re-loading) ---
bm25_model = None
sentence_transformer_model = None
document_chunks = [] # Store the indexed documents globally or pass them around

def load_embedding_model_at_startup():
    global sentence_transformer_model
    if sentence_transformer_model is None: # Only load if not already loaded
        print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME} at application startup...")
        sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Sentence Transformer model loaded.")

# --- Helper Functions ---

def process_markdown_with_manual_sections(
    md_file_path: str,
    headings_json: dict,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Processes a markdown document from a file path by segmenting it based on
    provided section headings, and then recursively chunking each segment.
    Each chunk receives the corresponding section heading as metadata.

    Args:
        md_file_path (str): The path to the input markdown (.md) file.
        headings_json (dict): A JSON object with schema: {"headings": ["Your Heading 1", "Your Heading 2"]}
                              This contains the major section headings to split by.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list[Document]: A list of LangChain Document objects, each containing
                              a text chunk and its associated metadata.
    """
    all_chunks_with_metadata = []
    full_text = ""

    # Check if the file exists and read its content
    if not os.path.exists(md_file_path):
        print(f"Error: File not found at '{md_file_path}'")
        return []
    if not os.path.isfile(md_file_path):
        print(f"Error: Path '{md_file_path}' is not a file.")
        return []
    if not md_file_path.lower().endswith(".md"):
        print(f"Warning: File '{md_file_path}' does not have a .md extension.")

    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except Exception as e:
        print(f"Error reading file '{md_file_path}': {e}")
        return []

    if not full_text:
        print("Input markdown file is empty.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Extract heading texts from the 'headings' key
    heading_texts = headings_json.get("headings", [])
    print(f"Identified headings for segmentation: {heading_texts}")

    # Find start indices of all headings in the full text using regex
    heading_positions = []
    for heading in heading_texts:
        # Create a regex pattern to match the heading, ignoring extra whitespace and making it case-insensitive
        pattern = re.compile(r'\s*'.join(re.escape(word) for word in heading.split()), re.IGNORECASE)
        
        match = pattern.search(full_text)
        if match:
            heading_positions.append({"heading_text": heading, "start_index": match.start()})
        else:
            print(f"Warning: Heading '{heading}' not found in the markdown text using regex. This section might be missed.")

    # Sort heading positions by their start index
    heading_positions.sort(key=lambda x: x["start_index"])

    # Segment the text based on heading positions
    segments_with_headings = []
    
    # Handle preface (text before the first heading)
    if heading_positions and heading_positions[0]["start_index"] > 0:
        preface_text = full_text[:heading_positions[0]["start_index"]].strip()
        if preface_text:
            segments_with_headings.append({
                "section_heading": "Document Start/Preface",
                "section_text": preface_text
            })

    # Iterate through heading positions to define sections
    for i, current_heading_info in enumerate(heading_positions):
        start_index = current_heading_info["start_index"]
        heading_text = current_heading_info["heading_text"]
        
        # Determine the end index for the current section
        end_index = len(full_text)
        if i + 1 < len(heading_positions):
            end_index = heading_positions[i+1]["start_index"]

        # Extract section content (from current heading's start to next heading's start)
        section_content = full_text[start_index:end_index].strip()
        
        if section_content:
            segments_with_headings.append({
                "section_heading": heading_text,
                "section_text": section_content
            })

    print(f"Created {len(segments_with_headings)} segments based on provided headings.")

    # Chunk each segment and attach metadata
    for segment in segments_with_headings:
        section_heading = segment["section_heading"]
        section_text = segment["section_text"]

        if section_text:
            chunks = text_splitter.split_text(section_text)
            for chunk in chunks:
                metadata = {
                    "document_part": "Section",
                    "section_heading": section_heading,
                }
                all_chunks_with_metadata.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"Created {len(all_chunks_with_metadata)} chunks with metadata from segmented sections.")
    
    # Not writing to output.json anymore as it was for evaluation of segmentation.
    with open("output.json", 'w', encoding='utf-8') as f:
        json.dump(segments_with_headings, f, indent=4, ensure_ascii=False)
    return all_chunks_with_metadata


def initialize_hybrid_search_models(documents: list[Document]):
    """
    Initializes BM25 and Sentence Transformer models.
    This function should be called once after documents are processed.
    """
    global bm25_model, sentence_transformer_model, document_chunks

    document_chunks = documents
    corpus = [doc.page_content for doc in documents]

    print("Initializing BM25 model...")
    tokenized_corpus = [doc.split(" ") for doc in corpus] # BM25 expects tokenized documents
    bm25_model = BM25Okapi(tokenized_corpus)
    print("BM25 model initialized.")

    print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
    sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Sentence Transformer model loaded.")

async def perform_hybrid_search(query: str, top_k: int) -> list[dict]:
    """
    Performs a hybrid search using BM25 and dense vectors (50-50 split).

    Args:
        query (str): The search query.
        top_k (int): The number of top relevant chunks to retrieve.

    Returns:
        list[dict]: A list of dictionaries, each containing 'content' and 'document_metadata'
                    from the combined search results.
    """
    if bm25_model is None or sentence_transformer_model is None or not document_chunks:
        raise ValueError("Hybrid search models are not initialized. Call initialize_hybrid_search_models first.")

    print(f"Performing hybrid search for query: '{query}' (top_k={top_k})...")

    # 1. BM25 (Sparse) Search
    tokenized_query = query.split(" ")
    bm25_scores = bm25_model.get_scores(tokenized_query)
    
    # 2. Dense Vector Search
    query_embedding = sentence_transformer_model.encode(query, convert_to_tensor=True)
    corpus_embeddings = sentence_transformer_model.encode([doc.page_content for doc in document_chunks], convert_to_tensor=True)
    
    # Compute cosine similarity
    from torch.nn.functional import cosine_similarity
    dense_scores = cosine_similarity(query_embedding, corpus_embeddings)
    dense_scores = dense_scores.cpu().numpy() # Convert back to numpy

    # 3. Normalize Scores
    # BM25 scores are typically non-negative, and can be large. Min-Max scaling is suitable.
    # Cosine similarity is already normalized between -1 and 1. We can scale it to 0-1.

    if len(bm25_scores) == 0 or len(dense_scores) == 0:
        return []

    scaler = MinMaxScaler()
    normalized_bm25_scores = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()

    # Scale dense scores to 0-1, if they are not already (cosine similarity is -1 to 1)
    # dense_scores_0_1 = (dense_scores + 1) / 2 # Already done by the time it is cosine_similarity
    normalized_dense_scores = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()

    # 4. Combine Scores (50-50 split)
    combined_scores = 0.5 * normalized_bm25_scores + 0.5 * normalized_dense_scores

    # 5. Get Top K Results
    ranked_indices = np.argsort(combined_scores)[::-1] # Sort in descending order
    top_k_indices = ranked_indices[:top_k]

    retrieved_results = []
    for idx in top_k_indices:
        doc = document_chunks[idx]
        retrieved_results.append({
            "content": doc.page_content,
            "document_metadata": doc.metadata
        })

    print(f"Retrieved {len(retrieved_results)} top chunks using hybrid search.")
    return retrieved_results


async def generate_answer_with_groq(query: str, retrieved_results: list[dict], groq_api_key: str) -> str:
    """
    Generates an answer using the Groq API based on the query and retrieved chunks' content.
    Includes metadata in the prompt for better context.

    Args:
        query (str): The original user query.
        retrieved_results (list[dict]): A list of dictionaries from search,
                                        each with 'content' and 'document_metadata'.
        groq_api_key (str): The Groq API key.

    Returns:
        str: The generated answer.
    """
    if not groq_api_key:
        return "Error: Groq API key is not set. Cannot generate answer."

    print("Generating answer with Groq API...")
    client = AsyncGroq(api_key= groq_api_key)

    context_parts = []
    for i, res in enumerate(retrieved_results):
        content = res.get("content", "")
        metadata = res.get("document_metadata", {})
        section_heading = metadata.get("section_heading", "N/A")
        document_part = metadata.get("document_part", "N/A")

        context_parts.append(
            f"--- Context Chunk {i+1} ---\n"
            f"Document Part: {document_part}\n"
            f"Section Heading: {section_heading}\n"
            f"Content: {content}\n"
            f"-------------------------"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a specialized document analyzer assistant. Your task is to answer the user's question "
        f"solely based on the provided context. Pay close attention to the section heading and document part "
        f"for each context chunk. Ensure your answer incorporates all relevant details, including any legal nuances "
        f"and conditions found in the context, and is concise, limited to one or two sentences. "
        f"Do not explicitly mention the retrieved chunks. If the answer cannot be found in the provided context, "
        f"clearly state that you do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.7,
            max_tokens=500,
        )
        answer = chat_completion.choices[0].message.content
        print("Answer generated successfully.")
        return answer
    except Exception as e:
        print(f"An error occurred during Groq API call: {e}")
        return "Could not generate an answer due to an API error."