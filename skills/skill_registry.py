
# ---------------------------------------------------------------------------
# AgentSkillRegistry — manages skills installed via `npx skills add`
# ---------------------------------------------------------------------------
from typing import List
from pathlib import Path


SKILLS_DIR = Path.home() / ".agents" / "skills"

class AgentSkillRegistry:
    """Manages agent skills installed in ~/.agents/skills."""

    def get_paths(self) -> List[str]:
        """Return directory paths for all installed agent skills.

        Pass to create_deep_agent(skills=...) or SkillsMiddleware(sources=...).
        """
        if not SKILLS_DIR.exists():
            return []
        return [
            str(p)
            for p in sorted(SKILLS_DIR.iterdir())
            if p.is_dir() and (p / "SKILL.md").exists()
        ]

    def list(self) -> List[str]:
        return [Path(p).name for p in self.get_paths()]

    def __repr__(self) -> str:
        return f"AgentSkillRegistry(skills={self.list()})"

