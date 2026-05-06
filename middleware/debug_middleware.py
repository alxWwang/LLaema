from typing import Any, Awaitable, Callable

from langchain_core.messages import ToolMessage, HumanMessage
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.runtime import Runtime

YELLOW = "\033[93m"
PINK   = "\033[95m"
RESET  = "\033[0m"


class DebugLoggingMiddleware(AgentMiddleware):
    """Prints agent input messages in yellow and tool outputs in pink."""

    def before_model(self, state: Any, runtime: Runtime) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        last = messages[-1] if messages else None
        if last is None:
            return
        if isinstance(last, HumanMessage):
            content = last.content if isinstance(last.content, str) else str(last.content)
            print(f"{YELLOW}[Input] {content}{RESET}")
        elif isinstance(last, ToolMessage):
            content = last.content if isinstance(last.content, str) else str(last.content)
            print(f"{YELLOW}[Tool Result → Model] {content[:200]}{RESET}")

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
    ) -> ToolMessage | Any:
        tool_name = request.tool_call.get("name", "?")
        tool_args = request.tool_call.get("args", {})
        print(f"{YELLOW}[Tool Call] {tool_name}({tool_args}){RESET}")
        result = await handler(request)
        if isinstance(result, ToolMessage):
            content = result.content if isinstance(result.content, str) else str(result.content)
            print(f"{PINK}[Tool Output] {tool_name}: {content[:300]}{RESET}")
        return result
