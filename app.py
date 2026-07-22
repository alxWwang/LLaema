
import asyncio
import json
from typing import Optional
from context.context import JarvisContextManager
from core.super_agent import Jarvis, MessageEvent
from skills.skill_hub import hub


async def main():
    tools = await hub.get_tools()
    context_manager = JarvisContextManager().mock_data()  # replace with JarvisContextManager().mock_data() if needef
    jarvis = Jarvis(tools=tools, debug=True, context_manager=context_manager)

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
