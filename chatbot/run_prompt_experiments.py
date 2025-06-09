# Jade Oakes
# May 1, 2025
# Runs prompt experiments with batched inference

import json
import os
import sys
import argparse
import torch
from multilingual_rag_chatbot_llm import retrieve_context, format_prompt, pipe
from prompt_loader import load_prompt_templates, get_prompt_by_id

# Define generation settings
GENERATION_SETTINGS = {
    "deterministic": {"do_sample": False},
    "balanced": {"do_sample": True, "top_p": 0.85, "temperature": 0.6}
}


def load_experiment_queries(path="chatbot/experiment_queries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Define queries
EXPERIMENT_QUERIES = load_experiment_queries()

# Define batch size
BATCH_SIZE = 2


def run_and_log_batch(batch, setting_name, setting_args, out_file, prompt_id):
    prompts = [entry["prompt"] for entry in batch]
    generations = pipe(prompts, max_new_tokens=512, batch_size=BATCH_SIZE, **setting_args)

    # Flatten output if needed
    if isinstance(generations[0], list):
        generations = [g for batch in generations for g in batch]


    for entry, result in zip(batch, generations):
        text = result['generated_text']
        final_answer = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()

        log_entry = {
            "mode": entry["mode"],
            "language": entry["lang"],
            "query": entry["query"],
            "setting": setting_name,
            "prompt_id": prompt_id,
            **setting_args,
            "context_used": entry["context_texts"],
            "prompt": entry["prompt"],
            "model_output": text,
            "final_answer": final_answer
        }
        out_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Free GPU memory after batch finishes
    torch.cuda.empty_cache()


def run_prompt_experiments(mode, prompt_id, output_path):
    prompt_templates = load_prompt_templates()
    prompt_instruction = get_prompt_by_id(prompt_templates, mode, prompt_id)
    if prompt_instruction is None:
        raise ValueError(f"Prompt ID '{prompt_id}' not found for mode '{mode}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Only run the experiment queries for the selected mode
        lang_dict = EXPERIMENT_QUERIES[mode]

        for lang, queries in lang_dict.items():
            for setting_name, setting_args in GENERATION_SETTINGS.items():
                batch = []
                for query in queries:
                    context = retrieve_context(query, k=5, source=mode)
                    context_texts = [c['text'] if mode == 'travel' else f"{c.get('en', '')} -> {c.get('es', '')}" for c in context]
                    prompt = format_prompt(query, context, source_mode=mode, instruction=prompt_instruction)

                    print(f"Queued: [{mode}] [{lang}] [{setting_name}] — {query}")

                    batch.append({
                        "mode": mode,
                        "lang": lang,
                        "query": query,
                        "context_texts": context_texts,
                        "prompt": prompt
                    })

                    if len(batch) == 4:
                        run_and_log_batch(batch, setting_name, setting_args, out_file, prompt_id)
                        batch = []
                if batch:
                    run_and_log_batch(batch, setting_name, setting_args, out_file, prompt_id)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    # If command-line arguments are passed, use argparse
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", required=True, help="Mode: general, travel, no_retrieval")
        parser.add_argument("--prompt_id", required=True, help="Prompt ID from prompt_templates.json")  # add _version# if needed
        parser.add_argument("--output", default="results/run_1/prompt_experiment_results.jsonl", help="Output file path")  # CHANGE FOLDER NAME FOR NEW RUN
        args = parser.parse_args()

        mode = args.mode
        prompt_id = args.prompt_id
        output_path = args.output

    else:
        # Manual defaults for running interactively
        print("No arguments passed — running in notebook/manual mode")
        mode = "general"
        prompt_id = "v1_general_default_en"
        output_path = f"results/run_1/manual_{mode}_{prompt_id}.jsonl"  # CHANGE FOLDER NAME FOR NEW RUN

    run_prompt_experiments(mode, prompt_id, output_path)
