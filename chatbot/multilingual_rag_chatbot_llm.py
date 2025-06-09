# Jade Oakes
# April 16, 2025
# Retrieves relevant sentence pairs from a FAISS index
# Uses an LLM to generate responses for translation, grammar, and info
# about different global cities

import os
import sys
import re
import json
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # LaBSE, MiniLM, distilUSE

# Load sentence pairs data FAISS index and metadata
index = faiss.read_index('data/sentence_pairs_index.faiss')  # add _version# if needed
with open('data/sentence_pairs_metadata.jsonl', 'r', encoding='utf-8') as f:  # add _version# if needed
    metadata = [json.loads(line) for line in f]

# Load travel data FAISS index and metadata
travel_index = faiss.read_index('data/chunked_travel_info_index.faiss')  # add _version# if needed
with open('data/chunked_travel_info_metadata.jsonl', 'r', encoding='utf-8') as f:  # add _version# if needed
    travel_metadata = [json.loads(line) for line in f]

# Authentication with Hugging Face
# Make sure to set your Hugging Face token in the environment as HF_TOKEN
# e.g., export HF_TOKEN=hf_xxx in your terminal, or use a .env file if supported
hf_token = os.environ["HF_TOKEN"]
login(hf_token)  # Hugging Face login

# Load LLM model
llm_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map='auto')

# Set pad token for batching and inference
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
llm.config.pad_token_id = tokenizer.pad_token_id

# Create pipeline
pipe = pipeline(
    'text-generation',
    model=llm,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.pad_token_id
)

# RAG helper
def retrieve_context(query, k=5, source="general"):
    if source == "no_retrieval":
        return []  # No context retrieved

    query_vector = model.encode([query], convert_to_numpy=True).astype('float32')

    if source == "travel":
        D, I = travel_index.search(query_vector, k)
        results = [travel_metadata[i] for i in I[0]]
    else:
        D, I = index.search(query_vector, k)
        results = [metadata[i] for i in I[0]]

    return results


def format_prompt(query, context, source_mode="general", instruction=None):
    # Build context block based on mode
    if source_mode == "no_retrieval":
        context_block = ""
    elif source_mode == "general":
        context_block = "\n".join([f"- {c['en']} (-> {c['es']})" for c in context])
    else:  # travel mode
        context_block = "\n".join([f"- {c['text']}" for c in context])

    # Use custom prompt instruction if provided
    if instruction:
        header = instruction.strip()
    else:
        # Default fallback instructions per mode
        if source_mode == "general":
            header = "You are a Spanish tutor and language partner. Help the user by translating, rephrasing, or continuing the conversation. Be helpful and encouraging."
        elif source_mode == "travel":
            header = "You are a friendly multilingual travel assistant. Provide helpful and accurate information about travel destinations, sightseeing, transportation, or local tips."
        else:
            header = "You are a helpful multilingual assistant. Answer the user's question naturally and informatively."

    # Format full prompt
    return f"""{header}

Context:
{context_block}

User: {query}
Answer:"""


def parse_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    mode_blocks = re.split(r'Mode:\s*(\w+)', text)[1:]
    parsed = []

    for i in range(0, len(mode_blocks), 2):
        mode = mode_blocks[i].strip().lower()
        block = mode_blocks[i + 1]

        en_queries = re.findall(r'En:\s*(.*?)\s*(Es:|$)', block, re.DOTALL)
        es_queries = re.findall(r'Es:\s*(.*)', block, re.DOTALL)

        en_lines = en_queries[0][0].strip().split('\n') if en_queries else []
        es_lines = es_queries[0].strip().split('\n') if es_queries else []

        for line in en_lines + es_lines:
            if line.strip():
                parsed.append((mode, line.strip()))

    return parsed


# Mode label mapping
mode_labels = {
    "general": "General Mode (translation, conversation)",
    "travel": "Travel Mode (destination info, sightseeing)",
    "no_retrieval": "No Retrieval Mode (baseline LLM response only)"
}

# CLI interaction
def run_cli():

    # BATCH MODE: run file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        queries = parse_queries(filepath)

        for mode, query in queries:
            print(f"\nMode: {mode_labels.get(mode, mode)}")
            print(f"Query: {query}")

            context = retrieve_context(query, k=5, source=mode)
            prompt = format_prompt(query, context, source_mode=mode)

            response = pipe(prompt, max_new_tokens=512)[0]['generated_text']
            answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()

            print(f"Answer: {answer}\n")

    else:
        print("Multilingual RAG Chatbot and Travel Assistant with LLM")
        print("-" * 30)
        print("Languages supported:")
        print("English\nSpanish\n")

        print("Modes:")
        print("1. General (translation, conversation)")
        print("2. Travel assistant (destinations, sightseeing, practical info)")
        print("3. No Retrieval (baseline LLM response only)\n")

        print("Type 'exit' to quit\n")

        # Initial mode selection
        while True:
            mode_input = input("Choose a mode (1, 2, or 3): ").strip()
            if mode_input == "1":
                source_mode = "general"
                break
            elif mode_input == "2":
                source_mode = "travel"
                break
            elif mode_input == "3":
                source_mode = "no_retrieval"
                break
            elif mode_input in ["exit", "quit"]:
                print("Goodbye!")
                exit()
            else:
                print("Please type 1, 2, 3, or 'exit'")

        print("Type '/switch' to change modes. Type 'exit' to quit \n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for using the chatbot!")
                break

            if user_input.lower() == "/switch":
                print("Switch to mode 1 (General), 2 (Travel), or 3 (No Retrieval)")
                while True:
                    new_mode = input("Mode: ").strip()
                    if new_mode in ["1", "2", "3"]:
                        if new_mode == "1":
                            source_mode = "general"
                        elif new_mode == "2":
                            source_mode = "travel"
                        else:
                            source_mode = "no_retrieval"
                        print(f"Mode switched to: {source_mode.title()}")

            # RAG section
            context = retrieve_context(user_input, k=5, source=source_mode)
            prompt = format_prompt(user_input, context, source_mode=source_mode)

            print("Generating answer...")

            # Dynamic tuning based on mode
            if source_mode == "travel":
                generation_kwargs = {
                    "do_sample": True,
                    "top_p": 0.85,
                    "temperature": 0.6
                }
            elif source_mode == "general":
                generation_kwargs = {
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.8
                }
            else:
                generation_kwargs = {
                    "do_sample": False
                }

            # Tune responses
            response = pipe(
                prompt,
                max_new_tokens=512,
                **generation_kwargs
                )[0]['generated_text']

            # Extract the final answer
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()

            print(f"\nBot: {answer}")

    print()


def generate_response(user_input, mode="general", instruction=None, do_sample=False, top_p=None, temperature=None):
    """
    Generates a response using the RAG pipeline or no-retrieval mode.

    Args:
        user_input (str): The user's question or message.
        mode (str): One of "general", "travel", or "no_retrieval".
        instruction (str, optional): Custom prompt instruction. Defaults to None.
        do_sample (bool): Whether to sample (creative mode). Defaults to False.
        top_p (float, optional): Top-p sampling parameter.
        temperature (float, optional): Temperature sampling parameter.

    Returns:
        str: Cleaned response string.
    """
    context = retrieve_context(user_input, k=5, source=mode)
    prompt = format_prompt(user_input, context, source_mode=mode, instruction=instruction)

    generation_args = {"max_new_tokens": 512, "do_sample": do_sample}
    if do_sample:
        if top_p is not None:
            generation_args["top_p"] = top_p
        if temperature is not None:
            generation_args["temperature"] = temperature

    response = pipe(prompt, **generation_args)[0]['generated_text']
    return response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()


# Only run CLI if directly invoked (not when imported by Streamlit)
if __name__ == "__main__":
    run_cli()