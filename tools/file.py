from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import os

link_metadata: List[Dict[str, str]] = []
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
file_vectorstore: FAISS = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def get_file_contents(file_path: str) -> str:
    """Reads the contents of a file."""
    with open(file_path, 'r') as f:
        return f.read()

def add_to_embedding_memory(file_path: str, title: str, content: str):
    """Splits content into chunks and adds them to the vector memory."""
    global file_vectorstore
    doc = Document(page_content=content, metadata={"file_path": file_path, "title": title})
    splits = text_splitter.split_documents([doc])
    if file_vectorstore is None:
        # Initialize FAISS with the first batch of documents
        file_vectorstore = FAISS.from_documents(splits, local_embeddings)
    else:
        file_vectorstore.add_documents(splits)
    print(f"[MEMORY] Processed {len(splits)} chunks from {file_path} into vectorstore.")