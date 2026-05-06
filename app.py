
import asyncio
import json
from context.context import JarvisContextManager
from core.super_agent import Jarvis
from skills.skill_hub import hub

async def main():
    cm = JarvisContextManager().mock_data()
    tools = await hub.get_tools()
    jarvis = Jarvis(tools=tools, context_manager=cm, debug=True)

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
