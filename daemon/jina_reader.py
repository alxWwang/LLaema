import torch
import requests
import re
import threading
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")

# Optimized Loading: Using bfloat16 to save 50% RAM
model = AutoModelForCausalLM.from_pretrained(
    "jinaai/ReaderLM-v2", 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True
).to(device)

def clean_html(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    junk_tags = ["script", "style", "meta", "link", "noscript", "header", "footer", "nav", "aside", "iframe", "svg"]
    for tag in soup(junk_tags):
        tag.decompose()
    
    main_section = soup.find("main") or soup.find("article")
    content_to_process = str(main_section) if main_section else (str(soup.body) if soup.body else str(soup))
    return " ".join(content_to_process.split())[:30000]

def create_prompt(text: str, tokenizer, instruction: str = None) -> str:
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    prompt = f"{instruction}\n```html\n{text}\n```"
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def stream_parse_html(url: str):
    # 1. Fetch and Clean
    html = requests.get(url, timeout=10).text
    cleaned_html = clean_html(html)
    input_prompt = create_prompt(cleaned_html, tokenizer=tokenizer)
    
    inputs = tokenizer(
        input_prompt, 
        return_tensors="pt", 
        padding=True, 
        return_attention_mask=True
    ).to(device)
    
    # 2. Extract both input_ids and attention_mask from the dictionary
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 3. Setup Streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 4. Pass the attention_mask into the generate call
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask, # <--- Pass it here
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0,
        do_sample=False,
        repetition_penalty=1.08,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id # Ensures the model knows which ID to use for padding
    )
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 5. Print tokens as they arrive
    print(f"\n--- Output for {url} ---\n")
    for new_text in streamer:
        print(new_text, end="", flush=True)
    
    thread.join()

# Example Usage
url = "https://www.nvidia.com/en-us/software/nvidia-app/"
stream_parse_html(url)