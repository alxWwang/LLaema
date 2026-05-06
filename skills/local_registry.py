# ---------------------------------------------------------------------------
# LocalRegistry — manages plain Python functions registered as LangChain tools
# ---------------------------------------------------------------------------

from typing import List, Optional, Callable
from langchain_core.tools import BaseTool, tool

class LocalRegistry:
    """Manages locally defined Python tools via the @skill decorator."""

    def __init__(self):
        self._tools: List[BaseTool] = []

    def skill(
        self,
        func: Optional[Callable] = None,
        *,
        requires_approval: bool = False,
    ):
        """Decorator that registers a function as a LangChain tool.

        @hub.local.skill
        def search_web(query: str) -> str: ...

        @hub.local.skill(requires_approval=True)
        def delete_file(path: str) -> str: ...
        """
        def decorator(f: Callable) -> BaseTool:
            t = tool(f) if not isinstance(f, BaseTool) else f
            t.metadata = t.metadata or {}
            t.metadata["requires_approval"] = requires_approval
            self._tools.append(t)
            return t

        if func is None:
            return decorator
        return decorator(func)

    def get_tools(self) -> List[BaseTool]:
        """Return all registered local tools."""
        return list(self._tools)

    def list(self) -> List[str]:
        return [t.name for t in self._tools]

    def __repr__(self) -> str:
        return f"LocalRegistry(tools={self.list()})"
