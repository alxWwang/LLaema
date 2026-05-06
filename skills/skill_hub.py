from typing import Any, Dict, List
from langchain_core.tools import BaseTool

from skills.MCP_registry import MCPRegistry
from skills.local_registry import LocalRegistry
from skills.skill_registry import AgentSkillRegistry

# ---------------------------------------------------------------------------
# SkillHub — unified entry point
# ---------------------------------------------------------------------------

class SkillHub:
    """Unified hub for all tools, MCP servers, and agent skills."""

    def __init__(self):
        self.mcp = MCPRegistry()
        self.local = LocalRegistry()
        self.agent_skills = AgentSkillRegistry()

    async def get_tools(self) -> List[BaseTool]:
        """Return all tools: local tools + MCP tools."""
        return self.local.get_tools() + await self.mcp.get_tools()

    def get_skill_paths(self) -> List[str]:
        """Return agent skill paths for create_deep_agent(skills=...)."""
        return self.agent_skills.get_paths()

    def list(self) -> Dict[str, Any]:
        return {
            "local": self.local.list(),
            "mcp": self.mcp.list(),
            "agent_skills": self.agent_skills.list(),
        }

    def __repr__(self) -> str:
        return (
            f"SkillHub(\n"
            f"  mcp={self.mcp},\n"
            f"  local={self.local},\n"
            f"  agent_skills={self.agent_skills}\n"
            f")"
        )


hub = SkillHub()
