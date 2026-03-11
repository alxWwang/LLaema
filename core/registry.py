
from typing import Dict, List
from langchain_core.tools import tool, BaseTool

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

@registry.register(requires_approval=True)
def get_file_contents(file_path: str) -> str:
    """Reads and returns the contents of a file."""
    from tools.memory import get_file_contents as _get_file_contents
    return _get_file_contents(file_path)

@registry.register(requires_approval=True)
def discover_new_links(query: str) -> str:
    """In a Graph Vector setup, your data isn't just a list of floating points. It’s a map where every point (Node) has a vector attached to it.

    Indexing: You take your data (e.g., a technical manual) and extract entities (e.g., "Mainboard," "CPU," "Voltage"). These become Nodes.

    Vectorization: Each node is turned into a vector so you can fi
    DISCOVERY MODE. Use this ONLY if you are looking for new information or URLs.
    Input must be a search query (e.g., 'latest news').
    """
    from tools.web import discover_new_links as _discover_new_links
    return _discover_new_links(query)

@registry.register(requires_approval=True)
def extract_full_content_from_url(url: str) -> str:
    """
    DEEP READ MODE. Use this ONLY when you have a specific URL.
    Input MUST be a URL starting with http:// or https://.
    This tool scrapes the full text and saves it to your vector memory.
    After calling this, use 'get_vector_memory' to query the saved content.
    """
    from tools.web import extract_full_content_from_url as _extract_full_content_from_url
    scrapped_content = _extract_full_content_from_url(url)
    return {
        "save_to_memory": True,
        "content": scrapped_content,
        "agent_message": f"Successfully scraped and saved content from {url} to memory. "
                         f"Use 'get_vector_memory' to query specific information from it.",
    }

@registry.register(requires_approval=True)
def get_vector_memory(query: str, topic: str = None) -> List[Dict[str, str]]:
    """
    SEARCH LONG-TERM MEMORY. 
    REQUIRED: Call this BEFORE 'search_web' if the user asks a follow-up question 
    about a URL you have already scraped.
    USE THIS FOR:
    - Recalling specific details from 'SUCCESSFULLY SCRAPED' pages.
    - Deep-diving into context that appeared in previous turns.
    - Answering 'What else did that site say?' without re-scraping.
    """
    from core.context import JarvisContextManager
    context_manager = JarvisContextManager._instance
    return context_manager.query_memory(query, topic=topic)