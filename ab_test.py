#!/usr/bin/env python3
"""ab_test.py

Simple A/B test runner for chatbot models.

- Loads queries from data/ab_queries.txt (one per line) or falls back to a small built-in list.
- Runs each query against primary and alt models (config via env vars).
- Measures latency and response length, captures responses to JSONL in ab_results/.
- Prints a small summary with average latency and sample diffs.

Usage:
    python3 ab_test.py

Environment variables used (or set in .env):
    OPENAI_API_KEY, OPENAI_MODEL, ALT_OPENAI_MODEL

"""
import os
import time
import json
from pathlib import Path
from statistics import mean

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Read config from env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALT_OPENAI_MODEL = os.getenv("ALT_OPENAI_MODEL", "gpt-5-mini")

# Setup paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma_data"
RESULTS_DIR = ROOT / "ab_results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / f"ab_results_{int(time.time())}.jsonl"

# load queries
queries_file = DATA_DIR / "ab_queries.txt"
if queries_file.exists():
    with queries_file.open("r", encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip()]
else:
    queries = [
        "How do I reset my password?",
        "Printer won't print, what can I do?",
        "How to connect to VPN on macOS?",
    ]

# Load docs & embeddings / vectorstore (reuse chatbot.py's approach)
# Simple loader: load .txt files under data/
docs = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        loader = TextLoader(DATA_DIR / filename)
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = splitter.split_documents(docs)

# Embeddings: use same import strategy as chatbot.py
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception:
    import importlib
    module = importlib.import_module("langchain_openai")
    OpenAIEmbeddings = getattr(module, "OpenAIEmbeddings")

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please set it or copy .env.example to .env.")

emb = OpenAIEmbeddings(openai_api_key=openai_key)

# ensure persist dir exists
PERSIST_DIR.mkdir(exist_ok=True)

# Try to load existing Chroma, else build
try:
    vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=emb)
    _ = vectorstore.similarity_search("__health_check__", k=1)
except Exception:
    vectorstore = Chroma.from_documents(texts, emb, persist_directory=str(PERSIST_DIR))
    vectorstore.persist()

# Build chains
primary_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=OPENAI_MODEL, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)
alt_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=ALT_OPENAI_MODEL, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Run tests and record
latencies_primary = []
latencies_alt = []
lengths_primary = []
lengths_alt = []

with OUT_FILE.open("w", encoding="utf-8") as out:
    for q in queries:
        print(f"Running query: {q}")
        record = {"query": q, "timestamp": time.time(), "results": {}}

        t0 = time.time()
        p_ans = primary_chain.run(q)
        t1 = time.time()
        lat_p = t1 - t0
        latencies_primary.append(lat_p)
        lengths_primary.append(len(p_ans))
        record["results"]["primary"] = {"model": OPENAI_MODEL, "answer": p_ans, "latency": lat_p, "length": len(p_ans)}

        t0 = time.time()
        a_ans = alt_chain.run(q)
        t1 = time.time()
        lat_a = t1 - t0
        latencies_alt.append(lat_a)
        lengths_alt.append(len(a_ans))
        record["results"]["alt"] = {"model": ALT_OPENAI_MODEL, "answer": a_ans, "latency": lat_a, "length": len(a_ans)}

        out.write(json.dumps(record, ensure_ascii=False) + "\n")

# Summary
summary = {
    "primary_model": OPENAI_MODEL,
    "alt_model": ALT_OPENAI_MODEL,
    "queries": len(queries),
    "primary_avg_latency": mean(latencies_primary) if latencies_primary else None,
    "alt_avg_latency": mean(latencies_alt) if latencies_alt else None,
    "primary_avg_length": mean(lengths_primary) if lengths_primary else None,
    "alt_avg_length": mean(lengths_alt) if lengths_alt else None,
    "output_file": str(OUT_FILE),
}

print("\nA/B test summary:")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Results written to {OUT_FILE}")
