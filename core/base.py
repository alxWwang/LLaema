from typing import List, Optional
from pydantic import BaseModel, Field

class StepOutput(BaseModel):
    """Structured output for a single execution step."""
    confirmation: str = Field(
        description="A brief confirmation of the action taken (e.g., 'File read successfully'). Do NOT include raw content."
    )
    summary: str = Field(
        description="A CONCISE summary of the result. Maximum 3 sentences. Strictly no fluff."
    )
    key_facts: List[str] = Field(
        description="A list of 1-3 key extracted facts or data points. Keep them atomic.",
        max_items=3
    )
    # Optional: A field for code if the user specifically asked for it
    code_snippet: Optional[str] = Field(
        description="Only populate this if the task explicitly asks to WRITE or SHOW code. Otherwise leave empty.",
        default=None
    )