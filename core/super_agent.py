import asyncio
import json
import os
import operator
from typing import TypedDict, Annotated, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv

from core.registry import registry
from middleware.debug_middleware import DebugLoggingMiddleware
from skills.skill_hub import hub

THREAD_ID = "jarvis-main"
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


class Jarvis:
    def __init__(self, tools: list = None, model: str = "claude-haiku-4-5-20251001", context_manager=None, debug: bool = False):
        self.debug = debug
        self.tools = (tools or []) + registry.get_tools()
        self.context_manager = context_manager

        api_keys = self._get_keys()
        self.llm = ChatAnthropic(model=model, api_key=api_keys["ANTHROPIC"])

        self.app = self._build_graph()

    def _create_middleware(self) -> list:
        return [
            DebugLoggingMiddleware(),
        ]

    def _build_graph(self):
        graph = StateGraph(AgentState)

        agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="You are Jarvis, a warm, professional, and efficient Research Assistant.",
            middleware=self._create_middleware(),
            skills=hub.get_skill_paths(),
            backend=LocalShellBackend(root_dir=os.getcwd(), virtual_mode=False, timeout=15, inherit_env=True),
        )

        async def topic_node(state: AgentState):
            messages = state["messages"]
            user_messages = [m for m in messages if isinstance(m, HumanMessage)]
            current_prompt = user_messages[-1].content if user_messages else ""

            topic_points = self.context_manager.topic_manager.identify_topic(current_prompt)

            if not topic_points:
                print(f"\033[93m[Context] No clear topic from memory. Using 'general'.\033[0m")
                self.context_manager.topic_manager.add_current_topic({})
            else:
                print(f"\033[93m[Context] Identified topic: '{list(topic_points.keys())}'\033[0m")
                self.context_manager.topic_manager.add_current_topic(topic_points)

            topics_str = " | ".join(f"{i}: {t}" for i, t in self.context_manager.topic_manager.current_topics.items())
            print(f"\033[93m[Context] | {topics_str} |\033[0m\n")
            return None

        async def agent_node(state: AgentState):
            input_len = len(state["messages"])
            result = await agent.ainvoke({"messages": state["messages"]})
            return {"messages": result["messages"][input_len:]}

        graph.add_node("topic", topic_node)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "topic")
        graph.add_edge("topic", "agent")
        graph.add_edge("agent", END)

        return graph.compile(checkpointer=MemorySaver())

    def _get_keys(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        return {"ANTHROPIC": key}

    async def single_run(self, user_input: str):
        config = {"configurable": {"thread_id": THREAD_ID}}
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        return result["messages"][-1].content

    def get_history(self) -> list:
        config = {"configurable": {"thread_id": THREAD_ID}}
        snapshot = self.app.get_state(config)
        return snapshot.values.get("messages", [])


async def main():
    tools = await hub.get_tools()
    jarvis = Jarvis(tools=tools, debug=True)

    print("\033[93m[Jarvis]\033[0m Online. Type 'quit' to exit.\n")

    while True:
        user_input = input("\033[96m[You]\033[0m ")
        if user_input.lower() == "quit":
            break
        response = await jarvis.single_run(user_input)
        print(f"\033[92m[Jarvis]\033[0m {response}\n")

    with open("conversation_history.json", "w") as f:
        json.dump(jarvis.get_history(), f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
