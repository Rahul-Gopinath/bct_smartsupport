#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: chatbot.py
Author: ACL | Gilbert
Date: 2025-10-18
Version: 0.3
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

# Ensure OpenAI API key is set in the environment
openai_key = os.getenv("OPENAI_API_KEY")
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

# Load or build vector store
if not rebuild_index:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print("Loaded existing vector store from disk.")
else:
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    print("Created new vector store and saved to disk.")
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    vectorstore.persist()

# Set up the RAG question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Interactive chat loop
print("Chatbot ready. Type your questions below. Enter 'exit' to quit.")
while True:
    query = input("Question: ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("Answer:", answer)
