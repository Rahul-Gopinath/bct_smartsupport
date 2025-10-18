import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import importlib
try:
    # Preferred: new LangChain location for OpenAIEmbeddings
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception:
    # Fallback: try dynamic import of alternative package name without a static import
    try:
        module = importlib.import_module("langchain_openai")
        OpenAIEmbeddings = getattr(module, "OpenAIEmbeddings")
    except Exception:
        raise ImportError(
            "Could not import OpenAIEmbeddings from 'langchain.embeddings.openai' or 'langchain_openai'. "
            "Install 'langchain' or 'langchain-openai'."
        )
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Load OpenAI API key from environment. Do NOT hardcode API keys in source.
# Try to load a local .env file if python-dotenv is available. This is optional.
try:
    from dotenv import load_dotenv
    # load .env from repo root (silent if file missing)
    load_dotenv()
except Exception:
    # python-dotenv not installed, continue — environment variables still work
    pass

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    # Warn but allow local experiments that don't call OpenAI to continue
    print("WARN: OPENAI_API_KEY not set. Set environment variable OPENAI_API_KEY to enable OpenAI embeddings.")

docs = []
for file in os.listdir("data"):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join("data", file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print("Chatbot bereit. Geben Sie 'exit' ein zum Beenden.")
while True:
    query = input("Frage: ")
    if query.lower() == "exit":
        break
    result = qa_chain.run(query)
    print("Antwort:", result)
