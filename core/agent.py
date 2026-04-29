import ast
import asyncio
import os
import operator
from typing import TypedDict, Annotated, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import trim_messages

import datetime

from load_dotenv import load_dotenv
load_dotenv()

# Make sure you have your custom modules in the same directory
from context import JarvisContextManager
import registry

MCP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_servers.json")

# 1. Standard State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class Jarvis:
    def __init__(self, tools: list, model: str = "claude-haiku-4-5", debug: bool = False):
        self.debug = debug
        self.tools = tools + registry.registry.get_tools()

        # Securely load API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it before running.")

        self.llm = ChatAnthropic(
            model=model, 
            api_key=api_key,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.context_manager = JarvisContextManager()

        self.memory = MemorySaver()
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        async def generate_topic_key(user_input: str) -> str:
            """Use the LLM to generate a short topic key from the user's message."""
            existing_topics = self.context_manager.get_topic_keys()
            topics_list = ", ".join(existing_topics) if existing_topics else "none"
            sys_msg = SystemMessage(content=(
                "You are a topic classifier. Given the user's message, respond with ONLY a short topic key "
                "(2-4 words, lowercase, no punctuation). This will be used as a category label.\n\n"
                f"Existing topics: [{topics_list}]\n\n"
                "If the message clearly fits an existing topic, return that exact topic. "
                "Otherwise, create a concise new topic key.\n\n"
                "Respond with ONLY the topic key, nothing else."
            ))
            response = await self.llm.ainvoke([sys_msg, HumanMessage(content=user_input)])
            return response.content.strip().lower()

        async def context_detection_node(state: AgentState):
            messages = state["messages"]
            user_messages = [m for m in messages if isinstance(m, HumanMessage)]
            current_prompt = user_messages[-1].content if user_messages else ""
            
            selected_topic = self.context_manager.identify_topic(current_prompt)

            if selected_topic == 'general':
                generated_topic = await generate_topic_key(current_prompt)
                print(f"\033[93m[Context] No clear topic from memory. LLM generated: '{generated_topic}'\033[0m")
                self.context_manager.current_topic = generated_topic
            else:
                print(f"\033[93m[Context] Identified topic from memory: '{selected_topic}'\033[0m")
                self.context_manager.current_topic = selected_topic
            
            return None # We don't update state here, just the context manager

        async def save_to_context_node(state: AgentState):
            """Inspect the latest ToolMessages for a 'save_to_memory' flag and persist."""
            messages = state["messages"]
            
            recent_tool_msgs: List[ToolMessage] = []
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    recent_tool_msgs.append(msg)
                else:
                    break
            
            for tool_msg in recent_tool_msgs:
                try:
                    tool_output = ast.literal_eval(tool_msg.content) if isinstance(tool_msg.content, str) else tool_msg.content
                    if isinstance(tool_output, dict) and tool_output.get("save_to_memory"):
                        content = str(tool_output.get("content", ""))
                        topic = self.context_manager.current_topic or "general"
                        
                        if content:
                            sys_msg = SystemMessage(content="Summarize the following content into a concise utterance.")
                            response = await self.llm.ainvoke([sys_msg, HumanMessage(content=content)])
                            utterance = response.content.strip()
                            
                            self.context_manager.add_to_utterance_memory([utterance], topic=topic)
                            self.context_manager.add_to_memory(content, topic)
                            print(f"\033[93m[Save] Persisted tool output to context under topic '{topic}'\033[0m")
                        
                        agent_msg = tool_output.get(
                            "agent_message",
                            f"Content saved to memory under topic '{topic}'."
                        )
                        tool_msg.content = str(agent_msg)
                except (ValueError, SyntaxError):
                    pass
            return {"messages": recent_tool_msgs}

        async def agent_node(state: AgentState):
            sys_prompt = (
                "You are Jarvis, a warm, professional, and efficient Research Assistant.\n\n"
                "### CRITICAL RULES ###\n"
                "- If a user provides a URL or asks to 'search', trigger the necessary tools.\n"
                "- Provide a helpful, natural language summary based on tool results."
            )
            sys_msg = SystemMessage(content=sys_prompt)
            
            # Keep context window manageable, but never send orphan ToolMessages.
            # Anthropic requires each tool_result to correspond to a prior tool_use.
            window_messages = state["messages"][-30:]
            cleaned_messages: List[BaseMessage] = []
            seen_tool_use_ids = set()

            for msg in window_messages:
                if isinstance(msg, AIMessage):
                    for tc in (msg.tool_calls or []):
                        tc_id = tc.get("id")
                        if tc_id:
                            seen_tool_use_ids.add(tc_id)
                    cleaned_messages.append(msg)
                elif isinstance(msg, ToolMessage):
                    if msg.tool_call_id in seen_tool_use_ids:
                        cleaned_messages.append(msg)
                else:
                    cleaned_messages.append(msg)

            recent_messages = cleaned_messages[-12:]
            messages = [sys_msg] + recent_messages
            
            if self.debug: print(f"\n\033[93m[Agent] Processing reasoning...\033[0m")

            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        async def tool_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_outputs = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                output = None
                approved = True
                
                print(f"\n\033[93m[MCP] Agent requested: {tool_name}({tool_args})\033[0m")
                
                if registry.registry.get_approval(tool_name):
                    approval = input("Execute? (y/n): ").strip().lower()
                    approved = True if approval == 'y' else False
                
                if approved:
                    try:
                        selected_tool = next((t for t in self.tools if t.name == tool_name), None)
                        if selected_tool:
                            output = await selected_tool.ainvoke(tool_args)
                        else:
                            output = f"Error: Tool {tool_name} not found."
                    except Exception as e:
                        output = f"Error executing tool: {e}"
                else:
                    output = "Tool execution denied by user."
                    
                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_id))
            
            return {"messages": tool_outputs}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        # Build the Graph
        workflow.add_node("context_detection", context_detection_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("save_to_context", save_to_context_node)

        # Flow: Context -> Agent -> (Tools -> Save -> Agent) -> END
        workflow.add_edge("context_detection", "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "save_to_context")
        workflow.add_edge("save_to_context", "agent")
        
        workflow.set_entry_point("context_detection")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.memory)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        async def agent_node(state: AgentState):    
            last_human_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
            print(f"\n\033[94m[User Input] {last_human_msg.content if last_human_msg else 'N/A'}\033[0m")
            
            sys_msg = SystemMessage(content= "You are Jarvis, a warm, professional, and efficient Research Assistant.\n\n"
                "The overall task was to respond to: {user_input}\n\n"
                "Continue working towards this goal")

            if self.debug:
                print(f"\n\033[93m[Agent] Processing reasoning...\033[0m")

            response = await self.llm_with_tools.ainvoke([sys_msg] + state["messages"][-10:])
            return {"messages": [response]}

        async def tool_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_outputs = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                output = None

                print(f"\n\033[93m[MCP] Agent requested: {tool_name}({tool_args})\033[0m")
                try:
                    selected_tool = next((t for t in self.tools if t.name == tool_name), None)
                    if selected_tool:
                        output = await selected_tool.ainvoke(tool_args)
                    else:
                        output = f"Error: Tool {tool_name} not found."
                except Exception as e:
                    output = f"Error executing tool: {e}"
                text_output = str(output)
                print(f"\033[93m[Tool Output] {text_output[:200]}\033[0m")
                token_count = self.llm.get_num_tokens(text_output)
                print(f"[Tokens] ~{token_count}")
                
                if token_count < 36000 and token_count > 8000:
                    print(f"\033[91m[Warning] Tool output exceeds token limit -> saving to rag: {token_count}\033[0m")
                    print(f"\033[91m Saving...\033[0m")
                    dt = datetime.datetime.now().isoformat()
                    self.context_manager.add_session_memory(text_output, id=f"{tool_id}_{dt}")
                    print(f"\033[91m Done\033[0m")
                    output = f"Tool output too long to return directly. Saved to session memory with ID: {tool_id}_{dt}"
                    
                elif token_count >= 36000:
                    print(f"\033[91m[Warning] Tool output exceeds absolute token limit: {token_count}. Not saving to memory.\033[0m")
                    output = f"Tool output exceeds absolute token limit and cannot be returned or saved. Try a smaller query or tool action."

                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_id))
            
            return {"messages": tool_outputs}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        # Build the Graph
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        workflow.set_entry_point("agent")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.memory)
    
    @staticmethod
    def extract_text_from_messages(content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            result = []

            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        result.append(part.get("text", ""))

            return "".join(result)

        return ""
        
    async def chat(self, user_input: str):
        print(f"\033[92m[Jarvis]\033[0m ", end="", flush=True)

        user_msg = HumanMessage(content=user_input)
        
        # Thread ID ensures LangGraph remembers the conversation history
        config = {"configurable": {"thread_id": "cli_session_1"}}
        
        # Only pass the new message; the checkpointer handles the rest
        initial_state = {"messages": [user_msg]}
        output_tokens = None

        async for chunk in self.app.astream({"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            message = chunk[0]
            if isinstance(message, AIMessage) and hasattr(message, "content") and message.content:
                print(self.extract_text_from_messages(message.content), end="", flush=True)

            usage_metadata = getattr(message, "usage_metadata", None) if isinstance(message, AIMessage) else None
            if usage_metadata and usage_metadata.get("output_tokens") is not None:
                output_tokens = usage_metadata["output_tokens"]

        print()
        if output_tokens is not None:
            print(f"\033[90m[Tokens] output={output_tokens}\033[0m")

async def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message="Key '\\$schema' is not supported")

    client = MultiServerMCPClient({
        "example_filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            "transport": "stdio"
        }
    })
    
    tools = await client.get_tools()
    agent = Jarvis(debug=False, tools=tools)

    print("\033[96mJarvis is online. Type 'exit' to quit.\033[0m\n" + "-"*40)

    while True:
        try:
            user_input = input(f"\n\033[96m[User]\033[0m ")
            if user_input.lower() in ["quit", "exit"]: 
                break
            await agent.chat(user_input)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())