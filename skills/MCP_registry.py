
# ---------------------------------------------------------------------------
# MCPRegistry — manages MCP server configs and exposes them as LangChain tools
# ---------------------------------------------------------------------------

import os
from typing import Any, Callable, Dict, List, Optional
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------------------------------------------------------------------
# MCP presets
# ---------------------------------------------------------------------------

_MCP_PRESETS: Dict[str, Callable[[Optional[Dict[str, str]]], Dict[str, Any]]] = {
    "filesystem": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "brave-search": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "transport": "stdio",
        "env": {"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", ""), **(env or {})},
    },
    "fetch": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "memory": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "sequential-thinking": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "puppeteer": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "github": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            **(env or {}),
        },
    },
    "slack": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "transport": "stdio",
        "env": {
            "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),
            "SLACK_TEAM_ID": os.environ.get("SLACK_TEAM_ID", ""),
            **(env or {}),
        },
    },
    "sqlite": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", (env or {}).pop("DB_PATH", "db.sqlite")],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "postgres": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", os.environ.get("POSTGRES_URL", "")],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
    "google-maps": lambda env: {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-google-maps"],
        "transport": "stdio",
        "env": {
            "GOOGLE_MAPS_API_KEY": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
            **(env or {}),
        },
    },
    "computer-use": lambda env: {
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-computer-use"],
        "transport": "stdio",
        **({"env": env} if env else {}),
    },
}



class MCPRegistry:
    """Manages MCP server installations."""

    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}

    def install(
        self,
        name: str,
        env: Optional[Dict[str, str]] = None,
        alias: Optional[str] = None,
    ) -> "MCPRegistry":
        """Install a named MCP preset (e.g. 'filesystem', 'brave-search')."""
        if name not in _MCP_PRESETS:
            available = ", ".join(sorted(_MCP_PRESETS))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        self._configs[alias or name] = _MCP_PRESETS[name](dict(env) if env else None)
        return self

    def install_raw(self, name: str, config: Dict[str, Any]) -> "MCPRegistry":
        """Install an arbitrary MCP server config dict."""
        self._configs[name] = config
        return self

    def uninstall(self, name: str) -> "MCPRegistry":
        """Remove an installed MCP server."""
        self._configs.pop(name, None)
        return self

    async def get_tools(self) -> List[BaseTool]:
        """Return LangChain tools from all installed MCP servers."""
        if not self._configs:
            return []
        client = MultiServerMCPClient(self._configs)
        return await client.get_tools()

    def list(self) -> Dict[str, Any]:
        return {
            "installed": list(self._configs.keys()),
            "available_presets": sorted(_MCP_PRESETS.keys()),
        }

    def __repr__(self) -> str:
        return f"MCPRegistry(installed={list(self._configs.keys())})"
