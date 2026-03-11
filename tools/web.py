import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

link_metadata: List[Dict[str, str]] = []
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
web_vectorstore: FAISS = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

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

def extract_full_content_from_url(url: str) -> str:
    """
    DEEP READ MODE. Use this ONLY when you have a specific URL.
    Input MUST be a URL starting with http:// or https://.
    This tool scrapes the full text and saves it to your vector memory.
    """
    is_url = url.lower().startswith(("http://", "https://"))
    if is_url:
        content, title =  scrape_link(url)
        output =  (f"SUCCESSFULLY SCRAPED {url}\n"
                f"TITLE: {title}\n"
                f"CONTENT: {content[:500]}...\n\n"
                )
        print(f"\033[95m{output}\033[0m")
        return output
    else:
        return "Error: Provided input is not a valid URL.Retry with a full URL starting with http:// or https://"

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
