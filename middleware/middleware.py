import uuid
from typing import Any, Awaitable, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from pydantic import BaseModel, Field

from context.context import JarvisContextManager


class _GetSessionMemoryInput(BaseModel):
    id: str = Field(description="The session memory ID returned when a tool output was too large.")


def _get_session_memory(id: str) -> dict:
    ctx = JarvisContextManager._instance
    if ctx is None:
        return {"error": "No context manager initialized."}
    return ctx.session_memory.get_session_memory(id)


async def _aget_session_memory(id: str) -> dict:
    return _get_session_memory(id)


class SessionMemoryMiddleware(AgentMiddleware):
    """Intercepts large tool outputs, stores them in session memory, and replaces
    the output with a reference ID and instructions to retrieve chunks via
    get_session_memory."""

    def __init__(self, max_chars: int = 6000):
        super().__init__()
        self.max_chars = max_chars
        self.tools = [
            StructuredTool.from_function(
                name="get_session_memory",
                description=(
                    "Retrieve one chunk of a large tool output stored in session memory. "
                    "Use this when a previous tool result was too large and was replaced "
                    "with a session memory ID. Returns the next chunk and the number of "
                    "remaining chunks. Keep calling with the same ID until chunk is null."
                ),
                func=_get_session_memory,
                coroutine=_aget_session_memory,
                args_schema=_GetSessionMemoryInput,
                infer_schema=False,
            )
        ]

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
    ) -> ToolMessage | Any:
        result = await handler(request)

        if request.tool_call.get("name") == "get_session_memory":
            return result

        if not isinstance(result, ToolMessage):
            return result

        content = result.content if isinstance(result.content, str) else str(result.content)

        if len(content) <= self.max_chars:
            return result

        ctx = JarvisContextManager._instance
        if ctx is None:
            return result

        session_id = f"sm_{request.tool_call['id']}_{uuid.uuid4().hex[:8]}"
        ctx.session_memory.add_session_memory(content, session_id)

        total_chunks = len(ctx.session_memory.session_literal_memory.get(session_id, []))
        replacement = (
            f"[OUTPUT STORED IN SESSION MEMORY]\n"
            f"The tool output was {len(content)} chars (too large for context).\n"
            f"Session ID: {session_id}\n"
            f"Total chunks: {total_chunks}\n"
            f"Call get_session_memory(id='{session_id}') to retrieve one chunk at a time. "
            f"Keep calling until the returned chunk is null."
        )

        return ToolMessage(
            content=replacement,
            tool_call_id=result.tool_call_id,
        )
