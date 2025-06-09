# Jade Oakes
# April 15, 2025
# Data preparation for multilingual chatbot with RAG
# Filters and compiles data from various sources into a single JSONL file
# Applies preprocessing and cleans the data

from langdetect import detect
from tqdm import tqdm
import json
import time
import random

def filter_tatoeba(
    infile,
    eng_outfile,
    spa_outfile,
    min_length=3,
    max_length=1000,
    remove_duplicates=True,
    use_langdetect=False,
    max_lines=None,
    sample_size=None,  # set to a number n to process n random lines from each file
    print_examples=False
):
    """Filter Tatoeba lines, each containing English and Spanish separated by tabs and write the filtered lines to a new file"""

    with open(infile, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    if max_lines:
        lines = lines[:max_lines]

    # Sample before filtering
    if sample_size and len(lines) > sample_size:
        lines = random.sample(lines, sample_size)

    filtered_pairs = []
    start_time = time.time()

    for line in tqdm(lines, desc="Filtering Tatoeba lines", ncols=100, unit="line"):
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 4:
            continue

        en_line = parts[1].strip()
        es_line = parts[3].strip()

        # Length checks
        if len(en_line) < min_length or len(es_line) < min_length:
            continue
        if len(en_line) > max_length or len(es_line) > max_length:
            continue

        # Language detection
        if use_langdetect:
            try:
                en_lang = detect(en_line)
                es_lang = detect(es_line)
                if en_lang != 'en' or es_lang != 'es':
                    continue
            except:
                continue

        # Add to the list if valid
        filtered_pairs.append((en_line, es_line))

    # Remove duplicates
    if remove_duplicates:
        filtered_pairs = list(set(filtered_pairs))

    if print_examples and filtered_pairs:
        print("Showing 5 filtered Tatoeba pairs:")
        for i, (en, es) in enumerate(filtered_pairs[:5]):
            print(f"Pair {i+1}:\n  [EN] {en}\n  [ES] {es}\n")

    with open(eng_outfile, 'w', encoding='utf-8') as f_en, open(spa_outfile, 'w', encoding='utf-8') as f_es:
        for en_line, es_line in filtered_pairs:
            f_en.write(en_line + "\n")
            f_es.write(es_line + "\n")

    kept = len(filtered_pairs)
    total = len(lines)
    percent = kept / total * 100 if total < 0 else 0

    print(f"Tatoeba filtering complete: {infile}")
    print(f"Kept {kept}/{total} lines after filtering ({percent:.1f}%).")
    print(f"Results saved to:\n  {eng_outfile}\n  {spa_outfile}")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")


def filter_parallel_corpus(
    eng_infile,
    spa_infile,
    eng_outfile,
    spa_outfile,
    min_length=3,
    max_length=1000,
    remove_duplicates=True,
    use_langdetect=False,
    max_lines=None,  # set to a number n to process only the first n lines from each file
    sample_size=None,  # set to a number n to process n random lines from each file
    print_examples=False  # set to True to print the first 5 pairs of lines
):
    """Filter parallel English-Spanish lines in sync and write to new files"""

    # Read lines
    with open(eng_infile, 'r', encoding='utf-8') as f_en, open(spa_infile, 'r', encoding='utf-8') as f_es:
        eng_lines = f_en.readlines()
        spa_lines = f_es.readlines()

    # Check line count
    if len(eng_lines) != len(spa_lines):
        raise ValueError(
            f"Mismatch in line count: {eng_infile} has {len(eng_lines)} lines, "
            f"{spa_infile} has {len(spa_lines)} lines"
        )

    # If max_lines is specified, slice the arrays
    if max_lines is not None:
        eng_lines = eng_lines[:max_lines]
        spa_lines = spa_lines[:max_lines]

    # Apply random sampling before filtering
    if sample_size and len(eng_lines) > sample_size:
        paired = list(zip(eng_lines, spa_lines))
        paired = random.sample(paired, sample_size)
        eng_lines, spa_lines = zip(*paired)

    total_lines = len(eng_lines)
    filtered_pairs = []
    start_time = time.time()

    # Wrap loop in tqdm for a progress bar
    for en_line, es_line in tqdm(zip(eng_lines, spa_lines), total=total_lines, desc="Filtering lines", dynamic_ncols=True, unit="line"):
        en_line = en_line.strip()
        es_line = es_line.strip()

        # Skip empty lines
        if not en_line or not es_line:
            continue

        # Length checks
        if len(en_line) < min_length or len(es_line) < min_length:
            continue
        if len(en_line) > max_length or len(es_line) > max_length:
            continue

        # Language detection
        if use_langdetect:
            try:
                en_lang = detect(en_line)
                es_lang = detect(es_line)
                # If lines do not match expected languages, skip
                if en_lang != 'en' or es_lang != 'es':
                    continue
            except:
                # If langdetect fails, skip the line
                continue

        # If it passes all filters, append it
        filtered_pairs.append((en_line, es_line))

    # Remove duplicates (optional)
    if remove_duplicates:
        filtered_pairs = list(set(filtered_pairs))

    # DEBUG: print examples if requested
    if print_examples and filtered_pairs:
        print("Showing up to 5 filtered line pairs:")
        for i, (en, es) in enumerate(filtered_pairs[:5]):
            print(f"Pair {i+1}:")
            print(f"  [EN]: {en}")
            print(f"  [ES]: {es}")
            print("")

    # Write to output
    with open(eng_outfile, 'w', encoding='utf-8') as f_en_out, open(spa_outfile, 'w', encoding='utf-8') as f_es_out:
        for en_line, es_line in filtered_pairs:
            f_en_out.write(en_line + "\n")
            f_es_out.write(es_line + "\n")

    kept = len(filtered_pairs)
    percent = kept / total_lines * 100 if total_lines < 0 else 0

    print(f"Filtering complete: {eng_infile} and {spa_infile}")
    print(f"Kept {kept}/{total_lines} lines after filtering ({percent:.1f}%)")
    print(f"Results saved to:\n  {eng_outfile}\n  {spa_outfile}")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")


def combined_filtered_files_to_jsonl(pairs_list, output_file):
    """
    Combine filtered data into one JSONL file
    Contains:
        English sentence
        Spanish sentence
        source
    """

    start_time = time.time()
    combined_data = []

    for en_path, es_path, source_name in pairs_list:
        with open(en_path, 'r', encoding='utf-8') as f_en, open(es_path, 'r', encoding='utf-8') as f_es:
            en_lines = f_en.readlines()
            es_lines = f_es.readlines()

        # Each source should have aligned line counts
        assert len(en_lines) == len(es_lines), f"Line mismatch in {source_name}"

        print(f"Processing {source_name}: {len(en_lines)} pairs")

        # Combine aligned lines from sources
        for en, es in tqdm(
        zip(en_lines, es_lines),
        total=len(en_lines),
        desc=f"Adding {source_name}",
        dynamic_ncols=True,
        unit="pair"
        ):
            en = en.strip()
            es = es.strip()
            combined_data.append({
                "en": en,
                "es": es,
                "source": source_name
            })

    # Write all combined entries to a single JSONL file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(
        combined_data,
        desc="Writing JSONL",
        dynamic_ncols=True,
        unit="entry"
        ):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print completion message and elapsed time
    elapsed = time.time() - start_time
    print(f"Combined file written to {output_file} with {len(combined_data)} entries")
    print(f"Elapsed time: {elapsed:.2f} seconds")

# CALL THE METHODS

# Load filenames
TATOEBA_INFILE = "data/tatoeba/tatoeba_en-es.tsv"
TATOEBA_ENG_OUTFILE = "data/tatoeba/tatoeba.en-es.en.filtered"
TATOEBA_SPA_OUTFILE = "data/tatoeba/tatoeba.en-es.es.filtered"
TATOEBA_ENG_300K_OUTFILE = "data/tatoeba/tatoeba_300k.en-es.en.filtered"
TATOEBA_SPA_300K_OUTFILE = "data/tatoeba/tatoeba_300k.en-es.es.filtered"

WIKIMATRIX_ENG_INFILE = "data/wikimatrix/wikimatrix.en-es.en"
WIKIMATRIX_SPA_INFILE = "data/wikimatrix/wikimatrix.en-es.es"
WIKIMATRIX_ENG_OUTFILE = "data/wikimatrix/wikimatrix.en-es.en.filtered"
WIKIMATRIX_SPA_OUTFILE = "data/wikimatrix/wikimatrix.en-es.es.filtered"
WIKIMATRIX_ENG_300K_OUTFILE = "data/wikimatrix/wikimatrix_300k.en-es.en.filtered"
WIKIMATRIX_SPA_300K_OUTFILE = "data/wikimatrix/wikimatrix_300k.en-es.es.filtered"

OPENSUBTITLES_ENG_INFILE = "data/opensubtitles/opensubtitles.en-es.en"
OPENSUBTITLES_SPA_INFILE = "data/opensubtitles/opensubtitles.en-es.es"
OPENSUBTITLES_ENG_OUTFILE = "data/opensubtitles/opensubtitles.en-es.en.filtered"
OPENSUBTITLES_SPA_OUTFILE = "data/opensubtitles/opensubtitles.en-es.es.filtered"
OPENSUBTITLES_ENG_300K_OUTFILE = "data/opensubtitles/opensubtitles_300k.en-es.en.filtered"
OPENSUBTITLES_SPA_300K_OUTFILE = "data/opensubtitles/opensubtitles_300k.en-es.es.filtered"

# FILTER DATA

# Tatoeba
print("TATOEBA")
print("-" * 100)
filter_tatoeba(
    infile=TATOEBA_INFILE,
    eng_outfile=TATOEBA_ENG_300K_OUTFILE,
    spa_outfile=TATOEBA_SPA_300K_OUTFILE,
    min_length=3,
    max_length=1000,
    remove_duplicates=True,
    use_langdetect=True,
    max_lines=None,  # set to a number n to process only the first n lines from each file
    sample_size=300000,  # 300k samples
    print_examples=True  # set to True to print the first 5 pairs of lines
)
print("DONE 1/3")
print()

# Wikimatrix
print("WIKIMATRIX")
print("-" * 100)
filter_parallel_corpus(
    eng_infile=WIKIMATRIX_ENG_INFILE,
    spa_infile=WIKIMATRIX_SPA_INFILE,
    eng_outfile=WIKIMATRIX_ENG_300K_OUTFILE,
    spa_outfile=WIKIMATRIX_SPA_300K_OUTFILE,
    min_length=3,
    max_length=1000,
    remove_duplicates=True,
    use_langdetect=True,
    max_lines=None,  # set to a number n to process only the first n lines from each file
    sample_size=300000, # 300k samples
    print_examples=True  # set to True to print the first 5 pairs of lines
)
print("DONE 2/3")
print()

# Open Subtitles
print("OPEN SUBTITLES")
print("-" * 100)
filter_parallel_corpus(
    eng_infile=OPENSUBTITLES_ENG_INFILE,
    spa_infile=OPENSUBTITLES_SPA_INFILE,
    eng_outfile=OPENSUBTITLES_ENG_300K_OUTFILE,
    spa_outfile=OPENSUBTITLES_SPA_300K_OUTFILE,
    min_length=3,
    max_length=1000,
    remove_duplicates=True,
    use_langdetect=True,
    max_lines=None,  # set to a number n to process only the first n lines from each file
    sample_size=600000,  # 600k samples
    print_examples=True  # set to True to print the first 5 pairs of lines
)
print("DONE 3/3")
print()

# COMBINE ALL FILTERED FILES INTO ONE JSONL

print("COMBINING FILTERED FILES INTO JSONL")
print("-" * 100)

combined_filtered_files_to_jsonl(
    pairs_list=[
        (TATOEBA_ENG_300K_OUTFILE, TATOEBA_SPA_300K_OUTFILE, "tatoeba"),
        (WIKIMATRIX_ENG_300K_OUTFILE, WIKIMATRIX_SPA_300K_OUTFILE, "wikimatrix"),
        (OPENSUBTITLES_ENG_300K_OUTFILE, OPENSUBTITLES_SPA_300K_OUTFILE, "opensubtitles")
    ],
    output_file="data/combined_300k_each.en-es.jsonl"  # add _version# if needed
)