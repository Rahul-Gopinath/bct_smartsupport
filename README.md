# bct_smartsupport
"""
Filename: chatbot.py
Author: ACL | Gilbert
Date: 2025-10-18
Version: 0.5
Description:
    This script implements a Retrieval-Augmented Generation (RAG) chatbot that uses
    local text documents as knowledge base and queries them with OpenAI's language 
    models. It persistently stores a vector index (Chroma) to speed up repeated runs 
    by avoiding re-processing documents unless explicitly requested.

Usage:
    python3 chatbot.py [-reindexing yes|no]

    Options:
    -reindexing yes  Rebuild the vector index from local documents, even if a saved index exists.
    -reindexing no   Load existing vector index if available, else build it (default).

A/B / Canary testing (optional):
    The script can run A/B or canary experiments between two generation models.
    Configure via environment variables (or a local `.env`):
      OPENAI_MODEL       - primary model name (default: gpt-4o-mini)
      ALT_OPENAI_MODEL   - alternative model name (e.g. gpt-5-mini)
      AB_MODE            - off | canary | both
                            - off: use primary model only
                            - canary: route a share of requests to ALT_OPENAI_MODEL
                            - both: run the query on both models and show both responses
      CANARY_PERCENT     - integer 0-100, percentage of requests routed to ALT model in canary mode

Requirements:
    - Python 3.9+
    - Set your OPENAI_API_KEY environment variable before running.
    - Install required packages: langchain, langchain-community, langchain-openai, etc.

This script handles:
    - Loading .txt documents from ./data/
    - Splitting documents into chunks for embedding
    - Creating or loading a persistent Chroma vector store
    - Using OpenAI embeddings and ChatOpenAI models from langchain
    - Command-line parameter parsing for flexible re-indexing

Note:
    For security, avoid hardcoding API keys; prefer environment variables.
    The script is designed for local MacOS usage but is cross-platform compatible.

"""