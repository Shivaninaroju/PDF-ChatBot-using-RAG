from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

# 1.Load PDF and extract text
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)  # Uses pypdf internally
    documents = loader.load()  # Loads each page as a document
    return documents

# 2.Split the text into smaller chunks (to avoid large token issues)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # each chunk = 500 characters
        chunk_overlap=50       # 50 characters repeated from previous to preserve context
    )
    chunks = splitter.split_documents(documents)
    return chunks

# 3.Create embeddings and store in FAISS vector DB
def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()  # Converts text to vectors using OpenAI
    vector_store = FAISS.from_documents(chunks, embeddings)  # Store vectors in FAISS
    return vector_store

# 4.Save FAISS DB locally (optional step for caching)
def save_vector_db(vector_store, path="faiss_index"):
    vector_store.save_local(path)

# 5.Load FAISS DB (if already saved)
def load_vector_db(path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(path, embeddings)
    return vector_store
