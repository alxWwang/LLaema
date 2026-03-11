import ast
import asyncio
import operator
from typing import TypedDict, Annotated, List

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from core.context import JarvisContextManager
import registry

# 1. Standard State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class Jarvis:
    def __init__(self, model: str = "ministral-3:14b", debug: bool = False):
        self.debug = debug
        self.tools = registry.registry.get_tools()
        
        # --- FIX 1: Initialize Persistent Memory ---
        self.chat_history: List[BaseMessage] = []
        
        self.llm = ChatOllama(model=model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.context_manager = JarvisContextManager()

        self.app = self._build_graph()


    def _build_graph(self):
        workflow = StateGraph(AgentState)

        def generate_topic_key(user_input: str) -> str:
            """Use the LLM to generate a short topic key from the user's message."""
            existing_topics = self.context_manager.get_topic_keys()
            topics_list = ", ".join(existing_topics) if existing_topics else "none"
            sys_msg = SystemMessage(content=(
                "You are a topic classifier. Given the user's message, respond with ONLY a short topic key "
                "(2-4 words, lowercase, no punctuation). This will be used as a category label.\n\n"
                f"Existing topics: [{topics_list}]\n\n"
                "If the message clearly fits an existing topic, return that exact topic. "
                "Otherwise, create a concise new topic key.\n\n"
                "Examples:\n"
                "- 'Tell me about quantum entanglement' → 'quantum computing'\n"
                "- 'How do I brew a V60?' → 'specialty coffee'\n"
                "- 'What is the weather today?' → 'weather'\n\n"
                "Respond with ONLY the topic key, nothing else."
            ))
            response = self.llm.invoke([sys_msg, HumanMessage(content=user_input)])
            return response.content.strip().lower()

        def context_detection_node(state: AgentState):
            messages = state["messages"]
            user_messages = [m for m in messages if isinstance(m, HumanMessage)]
            current_prompt = user_messages[-1].content if user_messages else ""
            selected_topic = self.context_manager.identify_topic(current_prompt)

            if selected_topic == 'general':
                generated_topic = generate_topic_key(current_prompt)
                print(f"\033[93m[Context] No clear topic from memory. LLM generated topic: '{generated_topic}'\033[0m")
                self.context_manager.current_topic = generated_topic
            else:
                print(f"\033[93m[Context] Identified topic from memory: '{selected_topic}'\033[0m")
                self.context_manager.current_topic = selected_topic

        def create_utterance_from_content(content: str) -> str:
            """Create a concise utterance from tool content for memory storage."""
            sys_msg = SystemMessage(content="Summarize the following content into a concise utterance that captures its essence. Be brief but informative.")
            messages = [sys_msg] + [HumanMessage(content=content)]
            response = self.llm.invoke(messages)
            return response.content.strip()
        
        def save_to_context_node(state: AgentState):
            """Inspect the latest ToolMessages for a 'save_to_memory' flag and persist to context."""
            messages = state["messages"]
            # Walk backwards to collect only the most recent contiguous batch of ToolMessages
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
                            utterance = create_utterance_from_content(content)
                            self.context_manager.add_to_utterance_memory([utterance], topic=topic)
                            self.context_manager.add_to_memory(content, topic)
                            print(f"\033[93m[Save] Persisted tool output to context under topic '{topic}'\033[0m")
                        # Replace the full content with a short confirmation for the agent

                        agent_msg = tool_output.get(
                            "agent_message",
                            f"Content saved to memory under topic '{topic}'. Use 'get_vector_memory' with topic='{topic}' to query it."
                        )
                        tool_msg.content = str(agent_msg)

                except (ValueError, SyntaxError):
                    pass
            return

        def agent_node(state: AgentState):
            sys_prompt = (
                "You are Jarvis, a warm, professional, and efficient Research Assistant.\n\n"
                
                "### TONE AND PERSONALITY ###\n"
                "- Always be friendly and conversational. Use emojis sparingly but warmly (😊).\n"
                "- Never output raw code, JSON, or tool syntax like 'search[ARGS]' directly to the user.\n\n"
                
                "### TOOL EXECUTION PROTOCOL ###\n"
                "1. THOUGHT: First, internally decide which tool you need. If the user asks about a previous topic, use 'get_vector_memory'. If it's new, use 'discover_new_links' and/or 'extract_full_content_from_url'.\n"
                "2. ACTION: Execute the tool using the formal tool-calling API. \n"
                "3. RESPONSE: Once you have the results, provide a helpful, natural language summary. Do NOT say 'I am now running a tool'. Just provide the answer found.\n\n"
                
                "### CRITICAL RULES ###\n"
                "- If a user provides a URL or asks to 'search' something, DO NOT try to answer from your own knowledge. You MUST trigger the 'discover_new_links' and/or 'extract_full_content_from_url' tool.\n"
                "- If you find yourself typing 'search[' or '{' in your response to the user, STOP. You are failing the protocol. Use the built-in tool-calling function instead."
            )
            sys_msg = SystemMessage(content=sys_prompt)
            
            # --- FIX 2: Context Window Management ---
            # We take the System Message + the last 10 messages from history
            # This keeps the prompt short for local models while retaining recent context.
            recent_messages = state["messages"][-10:] 
            messages = [sys_msg] + recent_messages

            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def tool_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_outputs = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                output = None
                approved = True
                print(f"\n\033[93m[SECURITY] Agent wants to run: {tool_name}({tool_args})\033[0m")
                if registry.registry.get_approval(tool_name):
                    approved = approval_request()
                if approved:
                    try:
                        selected_tool = next((t for t in self.tools if t.name == tool_name), None)
                        if selected_tool:
                            if self.debug: print(f"\033[90mRunning {tool_name}...\033[0m")
                            output = selected_tool.invoke(tool_args)
                        else:
                            output = f"Error: Tool {tool_name} not found."
                    except Exception as e:
                        output = f"Error executing tool: {e}"
                else:
                    output = "Tool execution denied by user."
                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_id))
            
            return {"messages": tool_outputs}

        def approval_request():
            approval = input("Execute? (y/n): ").strip().lower()
            return True if approval == 'y' else False
        
        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END
        

        # Add nodes
        workflow.add_node("context_detection", context_detection_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("save_to_context", save_to_context_node)

        # Add edges: agent -> tools -> save_to_context -> agent (loop)
        workflow.add_edge("context_detection", "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "save_to_context")
        workflow.add_edge("save_to_context", "agent")
        
        workflow.set_entry_point("context_detection")
        return workflow.compile()

    async def chat(self, user_input: str):
        print(f"\n\033[96m[User]\033[0m {user_input}")
        print(f"\033[92m[Jarvis]\033[0m ", end="", flush=True)

        # --- FIX 3: Add to History BEFORE Execution ---
        user_msg = HumanMessage(content=user_input)
        self.chat_history.append(user_msg)

        # Pass the ENTIRE history to the graph
        # The 'agent_node' will slice it down to the last 10 messages
        initial_state = {"messages": self.chat_history}

        final_response_text = ""

        async for event in self.app.astream_events(initial_state, version="v1"):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    final_response_text += chunk.content
        
        print() 

        # --- FIX 4: Update History with Agent's Reply ---
        # This ensures the next turn remembers what the agent said
        if final_response_text:
            self.chat_history.append(AIMessage(content=final_response_text))

async def main():
    agent = Jarvis(debug=True)
    while True:
        try:
            user_input = input("\n[User] ")
            if user_input.lower() in ["quit", "exit"]: break
            await agent.chat(user_input)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())