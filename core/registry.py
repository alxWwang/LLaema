
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
