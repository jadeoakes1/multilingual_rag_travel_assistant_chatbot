# Jade Oakes
# May 1, 2025
# Loads prompts from prompt_templates.json file

import json

def load_prompt_templates(path="chatbot/prompt_templates.json"):  # add _version# if needed
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_prompt_by_id(prompt_templates, mode, prompt_id):
    try:
        return prompt_templates[mode][prompt_id]
    except KeyError:
        raise ValueError(f"Prompt ID '{prompt_id}' not found for mode '{mode}'")