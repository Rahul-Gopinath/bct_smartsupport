#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import importlib

# Dynamic import for OpenAIEmbeddings (supporting multiple langchain versions)
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except ImportError:
    module = importlib.import_module("langchain_openai")
    OpenAIEmbeddings = getattr(module, "OpenAIEmbeddings")

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Try to load a local .env file if python-dotenv is available (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or .env missing — that's fine, environment variables still work
    pass

# Ensure OpenAI API key is set in the environment
openai_key = os.getenv("OPENAI_API_KEY")
# Print only presence status (do NOT print the key)
print("OPENAI_API_KEY set:", bool(openai_key))
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

# Argument parser for the command-line interface
parser = argparse.ArgumentParser(description="Start a RAG chatbot with optional re-indexing.")
parser.add_argument(
    "-reindexing",
    choices=["yes", "no"],
    default="no",
    help="Whether to rebuild the vector index from documents (yes/no). Defaults to 'no'."
)
args = parser.parse_args()
reindex_param = args.reindexing.lower() == "yes"

# Directory for persistent Chroma vector store
persist_dir = "chroma_data"

# Check if persistent index exists
persist_dir_exists = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

# Decide whether to rebuild index:
# Rebuild if directory missing or user requested reindexing
rebuild_index = not persist_dir_exists or reindex_param

# Load text documents from ./data folder
docs = []
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("data", filename))
        docs.extend(loader.load())

# Split documents into smaller chunks for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = splitter.split_documents(docs)

# Initialize OpenAI embeddings with explicit API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Ensure persist directory exists
os.makedirs(persist_dir, exist_ok=True)

# Load existing vectorstore if possible, otherwise build and persist
vectorstore = None
if not rebuild_index:
    try:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        # Try a lightweight query/load to validate the persisted store
        _ = vectorstore.similarity_search("__health_check__", k=1)
        print("Loaded existing vector store from disk.")
    except Exception as e:
        print(f"Warning: Failed to load existing vector store ({e}). Rebuilding index...")

if vectorstore is None:
    # Build new vector store from documents and persist
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print("Created new vector store and saved to disk.")

# Set up the RAG question-answering chain
# Model configuration and optional A/B testing
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALT_OPENAI_MODEL = os.getenv("ALT_OPENAI_MODEL", "gpt-5-mini")
AB_MODE = os.getenv("AB_MODE", "off").lower()  # off | canary | both
try:
    CANARY_PERCENT = int(os.getenv("CANARY_PERCENT", "10"))
except ValueError:
    CANARY_PERCENT = 10

# Build primary chain
primary_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=OPENAI_MODEL, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Build alt chain (only created when needed)
alt_chain = None
if AB_MODE in ("both", "canary"):
    alt_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=ALT_OPENAI_MODEL, temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

# Interactive chat loop
print("Chatbot ready. Type your questions below. Enter 'exit' to quit.")
while True:
    query = input("Question: ")
    if query.lower() == "exit":
        break
    # Decide routing based on AB_MODE
    import random

    if AB_MODE == "off":
        answer = primary_chain.run(query)
        print("Answer:", answer)

    elif AB_MODE == "canary":
        r = random.randint(1, 100)
        if r <= CANARY_PERCENT and alt_chain is not None:
            answer = alt_chain.run(query)
            print(f"[CANARY - {ALT_OPENAI_MODEL}] Answer:", answer)
        else:
            answer = primary_chain.run(query)
            print(f"[PRIMARY - {OPENAI_MODEL}] Answer:", answer)

    elif AB_MODE == "both":
        # Run both models (primary first) and print both outputs
        primary_answer = primary_chain.run(query)
        print(f"[PRIMARY - {OPENAI_MODEL}] Answer:", primary_answer)
        if alt_chain is not None:
            alt_answer = alt_chain.run(query)
            print(f"[ALT - {ALT_OPENAI_MODEL}] Answer:", alt_answer)
    else:
        # Unknown mode: fallback to primary
        answer = primary_chain.run(query)
        print("Answer:", answer)
