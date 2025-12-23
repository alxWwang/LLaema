from typing import Dict, List
from langchain_core.tools import tool, BaseTool
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ToolRegistry:
    def __init__(self):
        self._tools: List[BaseTool, bool] = []

    def register(self, func=None, *, requires_approval: bool = False):
        """Decorator to register a function as a LangChain tool."""
        def decorator(f):
            t = tool(f) if not isinstance(f, BaseTool) else f
            self._tools.append((t, requires_approval))
            return t

        if func is None:
            return decorator
        else:
            return decorator(func)

    def get_tools(self) -> List[BaseTool]:
        """Returns the list of tools for the LLM to bind."""
        return [t[0] for t in self._tools]

    def get_map(self) -> Dict[str, BaseTool]:
        """Returns a name->tool map for execution."""
        return {t[0].name: t[0] for t in self._tools}
    
    def get_approval(self, func_name) -> bool:
        """Returns approval requirement for a tool by name."""
        for t in self._tools:
            if t[0].name == func_name:
                return t[1]
        return False

registry = ToolRegistry()
link_metadata: List[Dict[str, str]] = []
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
global_vectorstore: FAISS = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@registry.register(requires_approval=False)
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@registry.register(requires_approval=False)
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b

@registry.register(requires_approval=False)
def get_super_secret(a: int, b: int) -> str:
    """Returns a super secret value if the user asks"""
    return "apple pen apple pen pie pie pineapple"

@registry.register(requires_approval=True)
def get_file_contents(file_path: str) -> str:
    """Reads and returns the contents of a file."""
    with open(file_path, 'r') as f:
        return f.read()
    
@registry.register(requires_approval=True)
def discover_new_links(query: str) -> str:
    """
    DISCOVERY MODE. Use this ONLY if you are looking for new information or URLs.
    Input must be a search query (e.g., 'latest news').
    """
    results = search_web(query)

    output = f"SEARCH RESULTS FOR: {query}\n\n"
    for r in results[:5]:
        url = r.get('url', '')
        title = r.get('title', '')
        snippet = r.get('content', '')
        
        # Save to Link Metadata (Short-term memory)
        link_metadata.append({"url": url, "title": title, "content": snippet})
        
        output += f"- {title}\n  URL: {url}\n  SNIPPET: {snippet}\n\n"
    
    output += "To read the full content of any link, use the 'extract_full_content_from_url' tool with the URL."
    print(f"\033[35m{output}\033[0m")
    return output

@registry.register(requires_approval=True)
def extract_full_content_from_url(url: str) -> str:
    """
    DEEP READ MODE. Use this ONLY when you have a specific URL.
    Input MUST be a URL starting with http:// or https://.
    This tool scrapes the full text and saves it to your vector memory.
    """
    is_url = url.lower().startswith(("http://", "https://"))
    if is_url:
        content, title =  scrape_link(url)

        # Save to Vector Memory
        add_to_embedding_memory(url, title, content)
        results = global_vectorstore.similarity_search(title, k=3)
        
        summary_bits = "\n".join([f"- {res.page_content}..." for res in results])

        output =  (f"SUCCESSFULLY SCRAPED {url}\n"
                f"TITLE: {title}\n"
                f"The full content is now in my long-term memory. "
                f"Initial relevant findings:\n{summary_bits}")
        print(f"\033[95m{output}\033[0m")
        return output
    else:
        return "Error: Provided input is not a valid URL.Retry with a full URL starting with http:// or https://"
    
@registry.register(requires_approval=True)
def get_vector_memory(query: str):
    """
    SEARCH INTERNAL LONG-TERM MEMORY. 
    REQUIRED: Call this BEFORE 'search_web' if the user asks a follow-up question 
    about a URL you have already scraped.
    
    USE THIS FOR:
    - Recalling specific details from 'SUCCESSFULLY SCRAPED' pages.
    - Deep-diving into context that appeared in previous turns.
    - Answering 'What else did that site say?' without re-scraping.
    """
    if global_vectorstore is None:
        return "No long-term memory initialized yet."
    docs = global_vectorstore.similarity_search(query, k=4)
    if not docs or (len(docs) == 1 and docs[0].metadata.get("url") == "init"):
        return "No specific details found in long-term memory regarding this query."
    formatted_results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("url", "Unknown Source")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content.strip()
        formatted_results.append(
            f"Result {i+1} [Source: {title} ({source})]:\n{content}"
        )
    return "\n\n---\n\n".join(formatted_results)

def add_to_embedding_memory(url: str, title: str, content: str):
    """Splits content into chunks and adds them to the vector memory."""
    global global_vectorstore
    doc = Document(page_content=content, metadata={"url": url, "title": title})
    splits = text_splitter.split_documents([doc])
    if global_vectorstore is None:
        # Initialize FAISS with the first batch of documents
        global_vectorstore = FAISS.from_documents(splits, local_embeddings)
    else:
        global_vectorstore.add_documents(splits)
    print(f"[MEMORY] Processed {len(splits)} chunks from {url} into vectorstore.")

def scrape_link(query: str) -> [str, str]:
    print(f"--- Action: Scraping Full URL via Jina ---")
    jina_url = f"https://r.jina.ai/{query}"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(jina_url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {})
        
        title = data.get("title", "No Title")
        content = data.get("content", "No content found.")

        return content, title
    except Exception as e:
        return f"Error scraping {query}: {e}", "Error"
    
def search_web(query: str) -> str:
    print(f"--- Action: Web Search for Snippets via SearXNG ---")
    searxng_url = "http://localhost:8888/searxng/search" # Ensure path is correct
    params = {"q": query, "format": "json"}
    try:
        response = requests.get(searxng_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as e:
        return f"Error contacting SearXNG: {e}", "Error"
    
    if not results:
        return "No results found."
    return results