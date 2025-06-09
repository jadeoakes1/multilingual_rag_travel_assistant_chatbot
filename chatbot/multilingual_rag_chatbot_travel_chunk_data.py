# Jade Oakes
# April 21, 2025
# Separates each JSONL file into chunks and combines all of the chunks
# into one JSONL file, which will be the metadata file read by the FAISS file

import os
import json
import nltk
import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

nltk.download("punkt")  # ensure tokenizer works

# Parameters
INPUT_FILE = "data/combined_travel_data.jsonl"  # add _version# if needed
OUTPUT_FILE = "data/chunked_travel_info_orig_data.jsonl"  # add _version# if needed

CHUNK_SIZE = 200  # words per chunk
MAX_TOKENS = 512  # token safety cap

# Split long text into 200-word chunks
def chunk_text(text, lang="en", max_words=CHUNK_SIZE, max_tokens=MAX_TOKENS):
    """Chunk text into ~200-word units, with token-length safety."""
    lang_nltk_map = {
        "en": "english",
        "es": "spanish"
    }
    nltk_lang = lang_nltk_map.get(lang)
    if not nltk_lang:
        raise ValueError(f"Unsupported language code: {lang}")

    # Normalize whitespace and remove invisible characters
    text = re.sub(r'\n+', '\n', text).replace('\xa0', ' ').strip()


    # Step 1: collect (sentence, section) tuples
    section = "intro"
    sentences_with_sections = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section headers like "== Eat =="
        section_match = re.match(r"^==+\s*(.*?)\s*==+$", line)
        if section_match:
            section = section_match.group(1).strip()
            continue

        if len(line.split()) > 20:
            sents = sent_tokenize(line, language=nltk_lang)
            for s in sents:
                # Extra clause splitting for flat lists
                if s.count(",") > 10 and s.count(".") < 2:
                    clauses = s.split(", ")
                    for c in clauses:
                        if len(c.split()) > 3:
                            sentences_with_sections.append((c, section))
                else:
                    sentences_with_sections.append((s, section))
        else:
            sentences_with_sections.append((line, section))


    # Step 2: chunk sentences and assign section
    chunks = []
    current_chunk = []
    current_word_count = 0
    current_section = None

    for sentence, sent_section in sentences_with_sections:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        if current_section is None:
            current_section = sent_section

        if current_word_count + sentence_len > max_words:
            chunk_text = " ".join(current_chunk)
            tokens = tokenizer.tokenize(chunk_text)
            if len(tokens) > max_tokens:
                for i in range(0, len(tokens), max_tokens):
                    sub_chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
                    chunks.append({"text": sub_chunk.replace("[UNK]", "").strip(), "section": current_section})
            else:
                chunks.append({"text": chunk_text.replace("[UNK]", "").strip(), "section": current_section})

            current_chunk = [sentence]
            current_word_count = sentence_len
            current_section = sent_section
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_len

    # Final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        tokens = tokenizer.tokenize(chunk_text)
        if len(tokens) > max_tokens:
            for i in range(0, len(tokens), max_tokens):
                sub_chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
                chunks.append({"text": sub_chunk.replace("[UNK]", "").strip(), "section": current_section})
        else:
            chunks.append({"text": chunk_text.replace("[UNK]", "").strip(), "section": current_section})

    # Filter short junk
    chunks = [c for c in chunks if len(tokenizer.tokenize(c["text"])) > 5]

    return chunks

# Combine all JSONL files with chunking
chunked_records = []

# Read from combined JSONL file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Chunking combined city data"):
        record = json.loads(line)
        city = record.get("city")
        lang = record.get("lang")
        text = record.get(lang)

        if not text:
            continue

        for i, chunk_data in enumerate(chunk_text(text, lang=lang)):
            chunked_records.append({
                "lang": lang,
                "city": city,
                "source": "wikivoyage",
                "chunk_id": f"{city.lower().replace(' ', '_')}_{lang}_{i}",
                "text": chunk_data["text"],
                "section": chunk_data["section"]
            })

# Save chunked data to combined output JSONL
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in chunked_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(chunked_records)} chunks to {OUTPUT_FILE}")