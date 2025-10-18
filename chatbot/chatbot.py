from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, NotFoundError
import numpy as np
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
INDEX_NAME = "pdf_chunks"
MAX_CHUNK_CHARS = 6000
CHUNK_OVERLAP = 200  # Characters to overlap between chunks
TOP_K_RESULTS = 3

# Validate API key
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

embeddings_model = OpenAIEmbeddings(api_key=API_KEY)

def make_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with overlap to preserve context.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    sentences = text.split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 2 < max_chars:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
                # Keep last 'overlap' characters for context
                words = current.split()
                overlap_text = " ".join(words[-(overlap // 5):])  # Rough word estimate
                current = overlap_text + sentence + ". "
            else:
                current = sentence + ". "

    if current:
        chunks.append(current.strip())

    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def connect_elasticsearch() -> Optional[Elasticsearch]:
    """Connect to Elasticsearch with error handling."""
    try:
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            logger.error("Cannot connect to Elasticsearch")
            return None
        logger.info("✅ Connected to Elasticsearch")
        return es
    except Exception as e:
        logger.error(f"Elasticsearch connection error: {e}")
        return None

def index_exists(es: Elasticsearch) -> bool:
    """Check if the index already exists."""
    try:
        return es.indices.exists(index=INDEX_NAME)
    except Exception as e:
        logger.error(f"Error checking index existence: {e}")
        return False

def create_vector_store(chunks: List[str], force_recreate: bool = False) -> Optional[Elasticsearch]:
    """
    Create or update Elasticsearch vector store.
    
    Args:
        chunks: List of text chunks to index
        force_recreate: If True, delete and recreate index even if it exists
    
    Returns:
        Elasticsearch client or None on failure
    """
    es = connect_elasticsearch()
    if not es:
        return None

    try:
        # Check if index exists
        if index_exists(es):
            if not force_recreate:
                logger.info(f"Index '{INDEX_NAME}' already exists. Skipping creation.")
                return es
            else:
                logger.info(f"Deleting existing index '{INDEX_NAME}'")
                es.indices.delete(index=INDEX_NAME)

        # Create index with mapping
        logger.info("Creating new index with embeddings...")
        sample_embedding = embeddings_model.embed_query(chunks[0])
        
        es.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": len(sample_embedding), "index": True, "similarity": "cosine"}
                }
            }
        )

        # Batch index chunks (process in batches to avoid rate limits)
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
            
            for j, chunk in enumerate(batch):
                try:
                    emb = embeddings_model.embed_query(chunk)
                    es.index(index=INDEX_NAME, id=i + j, document={"text": chunk, "embedding": emb})
                except Exception as e:
                    logger.error(f"Failed to index chunk {i + j}: {e}")

        logger.info(f"✅ Indexed {len(chunks)} chunks into Elasticsearch")
        return es

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def search_chunks(es: Elasticsearch, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
    """
    Search for relevant chunks using semantic similarity.
    
    Args:
        es: Elasticsearch client
        query: Search query
        top_k: Number of top results to return
    
    Returns:
        List of relevant text chunks
    """
    try:
        query_emb = embeddings_model.embed_query(query)

        resp = es.search(
            index=INDEX_NAME,
            size=top_k,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_emb}
                    }
                }
            }
        )

        results = [hit["_source"]["text"] for hit in resp["hits"]["hits"]]
        logger.info(f"Found {len(results)} relevant chunks")
        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def ask_question(es: Elasticsearch, query: str) -> str:
    """
    Answer a question using RAG.
    
    Args:
        es: Elasticsearch client
        query: User question
    
    Returns:
        Answer string
    """
    try:
        results = search_chunks(es, query, top_k=TOP_K_RESULTS)
        
        if not results:
            return "I couldn't find relevant information to answer your question."

        context = "\n\n".join(results)

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY, temperature=0)
        prompt = f"""You are a helpful assistant. Use only the following context to answer the question.
If the answer cannot be found in the context, reply "I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:"""

        # Correct invocation method
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return answer

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"An error occurred while processing your question: {str(e)}"

def interactive_mode(es: Elasticsearch):
    """Run chatbot in interactive mode."""
    print("\n🤖 Chatbot ready! Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            answer = ask_question(es, query)
            print(f"\n💬 Answer:\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")

if __name__ == "__main__":
    # Check if input file exists
    input_path = Path("output.txt")
    if not input_path.exists():
        logger.error(f"Input file '{input_path}' not found!")
        exit(1)

    # Load and chunk text
    try:
        text = input_path.read_text(encoding="utf-8")
        chunk_list = make_chunks(text)
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        exit(1)

    # Create vector store (only recreate if needed)
    es = create_vector_store(chunk_list, force_recreate=False)
    
    if not es:
        logger.error("Failed to create vector store. Exiting.")
        exit(1)

    # Interactive mode instead of single question
    interactive_mode(es)
