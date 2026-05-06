from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Dict


class SessionMemory():
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.session_literal_memory = {}

    def add_session_memory(self, content: str, id:str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=chunk, metadata={"id": id}) for chunk in chunks]
        print(f"[Session Memory] Adding content to session memory with ID: {id}, content length: {len(content)} Added {len(docs)} chunks.")
        self.session_literal_memory[id] = docs
    
    def get_session_memory(self, id: str) -> List[Dict]:
        chunk = self.session_literal_memory.get(id, [])[0].page_content if id in self.session_literal_memory and len(self.session_literal_memory[id]) > 0 else None
        self.session_literal_memory[id] = self.session_literal_memory[id][1:] if chunk else []
        print(f"[Session Memory] Retrieved chunk for ID: {id}, remaining chunks: {len(self.session_literal_memory.get(id, []))}")
        print(f"[Session Memory] Returning chunk content: {chunk[:200] if chunk else 'None'}")
        return {
            "length": len(self.session_literal_memory.get(id, [])),
            "chunk": chunk,
        }