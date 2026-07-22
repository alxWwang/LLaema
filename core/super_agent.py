import asyncio
import inspect
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Callable, Awaitable

from langchain.agents import create_agent

from langchain_openrouter import ChatOpenRouter
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv

from core.registry import registry
from middleware.debug_middleware import DebugLoggingMiddleware
from skills.skill_hub import hub

load_dotenv()

MessageEvent = dict  # {"id", "role", "content", "status", "timestamp"}
Subscriber = Callable[[MessageEvent], Awaitable[None] | None]


def _event(role: str, content: Optional[str], status: str, event_id: str = None) -> MessageEvent:
    return {
        "id": event_id or str(uuid.uuid4()),
        "role": role,
        "content": content,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class MessageStream:
    def __init__(self):
        self._subscribers: List[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Callable:
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    async def emit(self, event: MessageEvent):
        for cb in self._subscribers:
            if inspect.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)


class HistoryManager:
    def __init__(self):
        self.history: List[BaseMessage] = []
        self.subqueries: List[BaseMessage] = {}

        self.current_subquery_parent_id: Optional[str] = None

    def add(self, message: BaseMessage):
        self.history.append(message)
    
    def create_subquery(self, parent_id:str):
        self.subqueries[parent_id] = []
        self.current_subquery_parent_id = parent_id
    
    def add_subquery(self, message: BaseMessage):
        if self.current_subquery_parent_id:
            self.subqueries[self.current_subquery_parent_id].append(message)
        self.history.append(message)

    def update_query_result(self, parent_id: str, result: str):
        # find the original query in history and update its content with the result
        for i in range(len(self.history) - 1, -1, -1):
            if isinstance(self.history[i], AIMessage) and self.history[i].id == parent_id:
                self.history[i] = AIMessage(content=result, id=parent_id)
                break

    def update_subquery_result(self, parent_id: str, subquery_id: str, result: str):
        if parent_id in self.subqueries:
            for i in range(len(self.subqueries[parent_id]) - 1, -1, -1):
                if isinstance(self.subqueries[parent_id][i], AIMessage) and self.subqueries[parent_id][i].id == subquery_id:
                    self.subqueries[parent_id][i] = AIMessage(content=result, id=subquery_id)
                    break
    
    def merge_subqueries(self, parent_id: str):
        if parent_id in self.subqueries:
            # find the index of the original query in history
            parent_index = None
            for i in range(len(self.history) - 1, -1, -1):
                if isinstance(self.history[i], AIMessage) and self.history[i].id == parent_id:
                    parent_index = i
                    break
            if parent_index is not None:
                # insert subquery messages right after the original query
                self.history = self.history[:parent_index + 1] + self.subqueries[parent_id] + self.history[parent_index + 1:]
                del self.subqueries[parent_id]
                self.current_subquery_parent_id = None

    def get(self, recent_n: int = 20) -> List[BaseMessage]:
        return self.history[-recent_n:]

    def to_json(self) -> list:
        return [
            {
                "type": type(m).__name__,
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in self.history
        ]


class Jarvis:
    def __init__(self, tools: list = None, model: str = "deepseek/deepseek-v4-flash", context_manager=None, debug: bool = False):
        self.debug = debug
        self.tools = (tools or []) + registry.get_tools()
        self.context_manager = context_manager
        self.history_manager = HistoryManager()
        self.stream = MessageStream()
        self.running_task = False

        api_keys = self._get_keys()
        self.llm = ChatOpenRouter(model=model, api_key=api_keys["OPENROUTER"])
        self.agent_llm = ChatOpenRouter(model=model, api_key=api_keys["OPENROUTER_2"])

        self.agent = create_deep_agent(
            model=self.agent_llm,
            tools=self.tools,
            system_prompt="You are Jarvis, a warm, professional, and efficient Research Assistant.",
            middleware=[],
            skills=hub.get_skill_paths(),
            backend=LocalShellBackend(root_dir=os.getcwd(), virtual_mode=False, timeout=15, inherit_env=True),
        )

        self.topic_resolver = create_agent(
            model=self.llm,
            system_prompt="You are a pronoun resolver, given the task of identifying the pronouns in the last user message, and rewriting the prompt as if the pronouns were replaced with the actual subjects or objects",
        )

        self.quick_responder = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=(
                "You are a quick assistant standing in while the main agent is busy with a background task. "
                "The user has something running — a document being written, research being done, or a long task in progress. "
                "Your job is to answer helpful questions about anything else in the meantime using your general knowledge. "
                "If the question can be answered without tools or context from the ongoing task, answer it directly. "
            ),
        )

    def _get_keys(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        open_router_key = os.environ.get("OPENROUTER_KEY")
        open_router_key_2 = os.environ.get("OPENROUTER_KEY_2")
        if not open_router_key:
            raise ValueError("OPENROUTER_KEY environment variable not set.")
        if not open_router_key_2:
            raise ValueError("OPENROUTER_KEY_2 environment variable not set.")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        return {"ANTHROPIC": key, "OPENROUTER": open_router_key, "OPENROUTER_2": open_router_key_2}

    def _resolve_topic(self, messages: List[BaseMessage], current_prompt: str):
        if not self.context_manager:
            return
        prompt = f"""
        Identify the pronouns in THE_MESSAGE given the CHAT_HISTORY,
        and rewrite the message with pronouns replaced by the actual subjects or objects.
        IF there are no clear pronouns or subjects, or objects, return THE_MESSAGE as is without modification.
        DO NOT add any additional information that is not present in the original message or chat history.
        ONLY resolve pronouns that are clearly identifiable from the chat history.
        If a pronoun could refer to multiple subjects or objects, do not resolve it RETURN THE_MESSAGE.

        the users pronouns (me, I, my, mine) refer to the user, and should be resolved to USER.
        the agents pronouns (you, your, yours) refer to the agent, and should be resolved to JARVIS.

        THE_MESSAGE: {current_prompt}
        CHAT_HISTORY: {" | ".join(f"{type(m).__name__}: {m.content}" for m in messages[-5:])}
        """
        result = self.topic_resolver.invoke({"messages": [HumanMessage(content=prompt)]})
        topic_points = self.context_manager.topic_manager.identify_topic(result["messages"][-1].content)

        if not topic_points:
            self.context_manager.topic_manager.add_current_topic({})
        else:
            self.context_manager.topic_manager.add_current_topic(topic_points)

    async def _quick_check(self, snapshot: List[BaseMessage], event_id:str) -> Optional[str]:
        user_input = snapshot[-1].content if snapshot else ""
        if len(user_input) > 100:
            return None
        prompt = f"""
        Given the user message: {user_input}
        IF the message is a simple question or request that can be answered directly with general knowledge, return a helpful response.
        """
        messages = snapshot[:-1] + [HumanMessage(content=prompt)]
        response = await asyncio.to_thread(self.quick_responder.invoke, {"messages": messages})
        content = response["messages"][-1].content.strip()
        return content

    async def _run_agent(self, snapshot: List[BaseMessage], event_id: str):
        try:
            result = await asyncio.to_thread(self.agent.invoke, {"messages": snapshot})
            new_messages = result["messages"][len(snapshot):]
            if not new_messages:
                return

            ai_msg = new_messages[-1]
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

            # replace placeholder in history with real response
            for i in range(len(self.history_manager.history) - 1, -1, -1):
                if isinstance(self.history_manager.history[i], AIMessage) and \
                        self.history_manager.history[i].id == event_id:
                    self.history_manager.history[i] = AIMessage(content=content, id=event_id)
                    break

            await self.stream.emit(_event("assistant", content, "complete", event_id))
        finally:
            self.running_task = False

    async def handle_message(self, user_input: str):
        human_id = str(uuid.uuid4())
        user_message = HumanMessage(content=user_input, id=human_id)
        self.history_manager.add(user_message)
        snapshot = self.history_manager.get()

        await self.stream.emit(_event("human", user_input, "complete", human_id))
        # self._resolve_topic(snapshot, user_input)
        query_type = "query" if not self.running_task else "subquery"

        if query_type == "subquery":
            ai_id = str(uuid.uuid4())
            self.history_manager.add_subquery(AIMessage(content="Quick Thinking ...", id=ai_id))
            await self.stream.emit(_event("assistant", "Quick Thinking ...", "pending", ai_id))
            asyncio.create_task(self._quick_check(snapshot, ai_id))
            return

        if query_type == "query":
            ai_id = str(uuid.uuid4())
            self.history_manager.add(AIMessage(content="Processing...", id=ai_id))
            self.history_manager.create_subquery(ai_id)
            await self.stream.emit(_event("assistant", "Processing...", "pending", ai_id))
            self.running_task = True
            asyncio.create_task(self._run_agent(snapshot, ai_id))


async def main():
    tools = await hub.get_tools()
    jarvis = Jarvis(tools=tools, debug=True)

    def on_message(event: MessageEvent):
        role = event["role"]
        content = event["content"]
        status = event["status"]
        if role == "human":
            print(f"\033[96m[You]\033[0m {content}")
        elif role == "assistant" and status == "pending":
            print(f"\033[92m[Jarvis]\033[0m {content}\n")
        elif role == "assistant" and status == "complete":
            print(f"\033[92m[Jarvis]\033[0m {content}\n")

    jarvis.stream.subscribe(on_message)

    print("\033[93m[Jarvis]\033[0m Online. Type 'quit' to exit.\n")

    while True:
        user_input = await asyncio.to_thread(input, "")
        if user_input.lower() == "quit":
            break
        await jarvis.handle_message(user_input)

    with open("conversation_history.json", "w") as f:
        json.dump(jarvis.history_manager.to_json(), f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
