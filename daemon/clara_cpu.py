import os
import time
import hashlib
from typing import List, TypedDict
from glob import glob
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
INPUT_PATH = "/home/wang/.llaema_sample_input_path"       # Drop your raw PDFs/Text here
PROXIES_PATH = "/home/wang/.llaema_proxies"   # Clean summaries appear here
MODEL_NAME = "qwen2.5:3b-Instruct"            # Fast, low-resource model

class DaemonState(TypedDict):
    file_path: str
    raw_content: str
    summary: str

class BackgroundMemoryService:
    def __init__(self):
        print(f"⚙️  Starting Background Memory Service ({MODEL_NAME})...")
        print(f"    Watching: {INPUT_PATH}")
        print(f"    Output:   {PROXIES_PATH}")
        
        # 1. Initialize the specific "Compressor" model
        #    Temperature 0 forces it to be a cold, hard fact-extractor.
        self.llm = ChatOllama(
            model=MODEL_NAME, 
            temperature=0, 
        )
        
        os.makedirs(PROXIES_PATH, exist_ok=True)
        self.app = self.__build_graph()
    
    def __compression_ratio(self, original: str, summary: str) -> float:
        if len(original) == 0:
            return 0.0
        return len(summary) / len(original)

    def __build_graph(self):
        # --- NODE: COMPRESSOR ---
        def compress_node(state: DaemonState):
            filename = os.path.basename(state["file_path"])
            print(f"    [~] Compressing: {filename}...")
            
            system_msg = """You are a Universal Data Distillation Engine. 
                        Your primary function is to strip formatting noise while preserving 100% of the semantic logic and hard data.

                        ### GLOBAL CONSTRAINTS:
                        1.  **NO HALLUCINATIONS:** If a specific datum (date, name, ID) is not in the text, write "N/A". Never guess.
                        2.  **PRESERVE HIERARCHY:** If the input is a list, output a list. If it is a timeline, output a timeline.
                        3.  **VERBATIM ENTITIES:** Never summarize specific values. Copy-paste IDs, error codes, function names, and financial figures exactly.
                        4.  **OBJECTIVITY:** Remove all "fluff" (greetings, marketing adjectives, moralizing). Keep only the facts.
                        """

                        # HUMAN: DEFINES THE "OUTPUT SCHEMA"
            human_msg = """
                <SOURCE_DOCUMENT>
                {raw_content}
                </SOURCE_DOCUMENT>

                ### TASK: Convert the source above into the following Structured Knowledge Block.

                [SECTION 1: METADATA]
                * **Document_Type**: (Detect the format: e.g., Source Code, Email, Legal Contract, Research Paper, Log File)
                * **Key_Entities**: (List proper nouns, project names, authors, or server names)
                * **Timeline**: (List all extracted dates, timestamps, or durations)

                [SECTION 2: LOGICAL_SEGMENTS]
                (Instruction: Divide the text into logical blocks based on its content. Use generic headers like [PROBLEM], [SOLUTION], [METHOD], [CLAUSE], or [FUNCTION].)
                * [Header Name]
                    * **Core_Info**: (The main point or action item)
                    * **Details**: (Nuances, conditions, or arguments)

                [SECTION 3: HARD_DATA_EXTRACT]
                (Instruction: List specific values found. Do not summarize.)
                * **Metrics/Values**: (Any number with a unit, e.g., "50ms", "$10k", "Version 2.1")
                * **Technical_Refs**: (File paths, function names, error codes, API endpoints)
                * **Status_Flags**: (e.g., "DEPRECATED", "CONFIDENTIAL", "TODO", "APPROVED")

                [SECTION 4: QUALITY_CONTROL]
                * **Ambiguities**: (Is anything unclear or contradictory in the source?)
                * **Missing_Context**: (Does the text refer to attachments or previous files not present here?)
                """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("human", human_msg)
            ])
                        
            chain = prompt | self.llm | StrOutputParser()
            try:
                summary = chain.invoke({"raw_content": state["raw_content"]})
                ratio = self.__compression_ratio(state["raw_content"], summary)
                print(f"    [+] Compression complete: {filename}, compression ratio: {ratio:.2f}")
                print(f"    [+] Summary Preview: {summary[:75].replace(chr(10), ' ')}...")
                return {"summary": summary}
            except Exception as e:
                print(f"    [!] Error processing {filename}: {e}")
                return {"summary": ""}
            

        # --- NODE: SAVE IO ---
        def save_node(state: DaemonState):
            # Create a corresponding filename in the proxy folder
            base_name = os.path.basename(state["file_path"])
            save_name = f"{os.path.splitext(base_name)[0]}_proxy.txt"
            save_path = os.path.join(PROXIES_PATH, save_name)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(state["summary"])
            
            print(f"    [+] Saved Proxy: {save_name}")
            return state

        # Build Graph
        workflow = StateGraph(DaemonState)
        workflow.add_node("compress", compress_node)
        workflow.add_node("save", save_node)
        
        workflow.set_entry_point("compress")
        workflow.add_edge("compress", "save")
        workflow.add_edge("save", END)
        
        return workflow.compile()

    def scan_and_process(self):
        """Scans input folder and processes any file that doesn't have a proxy."""
        # Get list of raw files
        # Recursively crawl INPUT_PATH for all .txt files (expand to other types as needed)
        allowed_extensions = [".txt"]
        raw_files = []
        for root, dirs, files in os.walk(INPUT_PATH):
            for file in files:
                if any(file.lower().endswith(ext) for ext in allowed_extensions):
                    raw_files.append(os.path.join(root, file))
        # You can add support for other file types (e.g., .pdf) by extending the condition above
        
        processed_count = 0
        
        for file_path in raw_files:
            base_name = os.path.basename(file_path)
            proxy_name = f"{os.path.splitext(base_name)[0]}_proxy.txt"
            proxy_path = os.path.join(PROXIES_PATH, proxy_name)
            
            # Skip if proxy exists and is newer than raw file
            if os.path.exists(proxy_path):
                if os.path.getmtime(proxy_path) > os.path.getmtime(file_path):
                    continue
            
            # Read Raw Content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            if not content.strip():
                continue

            # Run Graph
            self.app.invoke({
                "file_path": file_path,
                "raw_content": content,
                "summary": ""
            })
            processed_count += 1
            
        if processed_count == 0:
            pass # Silent when idle
        else:
            print(f"    [=] Batch complete. Processed {processed_count} files.")

    def run_daemon(self, interval=10):
        print("🟢 Daemon Active. Waiting for files...")
        try:
            while True:
                self.scan_and_process()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🔴 Daemon Stopped.")

# --- USAGE ---
if __name__ == "__main__":
    # Ensure directories exist for the demo
    os.makedirs(INPUT_PATH, exist_ok=True)
    service = BackgroundMemoryService()
    service.run_daemon(interval=5)