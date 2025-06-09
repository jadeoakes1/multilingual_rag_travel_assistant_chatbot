# April 29, 2025
# Displays statistics for the data used in the RAG pipeline (both raw and metadata)

import json
import os
from tqdm import tqdm
from collections import defaultdict

def count_tokens(text):
    """Counts number of words (tokens separated by spaces)"""
    return len(text.split())

def analyze_jsonl(file_path, file_label, config):
    """Analyze a JSONL file for sample count, token count, unique documents, unique classes."""
    print(f"\nAnalyzing {file_label}: {file_path}")
    total_samples = 0
    total_tokens = 0
    doc_ids = set()
    categories = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=file_label, unit="lines"):
            item = json.loads(line)
            total_samples += 1

            # Count tokens
            if config["text_fields"]:
                for field in config["text_fields"]:
                    if field in item:
                        total_tokens += count_tokens(item[field])

            # Track unique documents and classes
            if config["doc_field"] and config["doc_field"] in item:
                doc_ids.add(item[config["doc_field"]])
            if config["class_field"] and config["class_field"] in item:
                categories.add(item[config["class_field"]])

    print(f"Samples: {total_samples}")
    print(f"Tokens: {total_tokens}")
    print(f"Unique Documents: {len(doc_ids)}")
    print(f"Unique Classes/Categories: {len(categories)}")

    return {
        "label": file_label,
        "samples": total_samples,
        "tokens": total_tokens,
        "documents": len(doc_ids),
        "classes": len(categories)
    }

def analyze_folder(folder_path, file_label, config):
    """Analyze a folder of JSONL files (for original wikivoyage)"""
    print(f"\nAnalyzing folder: {folder_path} ({file_label})")
    total_samples = 0
    total_tokens = 0
    doc_ids = set()
    categories = set()

    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    for file in tqdm(jsonl_files, desc=f"{file_label} files", unit="file"):
        path = os.path.join(folder_path, file)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                total_samples += 1

                if config["text_fields"]:
                    for field in config["text_fields"]:
                        if field in item:
                            total_tokens += count_tokens(item[field])
                if config["doc_field"] and config["doc_field"] in item:
                    doc_ids.add(item[config["doc_field"]])
                if config["class_field"] and config["class_field"] in item:
                    categories.add(item[config["class_field"]])

    print(f"Samples: {total_samples}")
    print(f"Tokens: {total_tokens}")
    print(f"Unique Documents: {len(doc_ids)}")
    print(f"Unique Classes/Categories: {len(categories)}")

    return {
        "label": file_label,
        "samples": total_samples,
        "tokens": total_tokens,
        "documents": len(doc_ids),
        "classes": len(categories)
    }

def analyze_by_source(file_path):
    """Break down a JSONL file by source field (for sentence pairs)"""
    print(f"\nBreakdown by source in {file_path}")
    source_counts = defaultdict(lambda: {"samples": 0, "tokens": 0})

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing sentence pairs"):
            item = json.loads(line)
            src = item.get("source", "unknown")
            source_counts[src]["samples"] += 1
            source_counts[src]["tokens"] += len(item.get("en", "").split())
            source_counts[src]["tokens"] += len(item.get("es", "").split())

    # Print results
    print(f"\n{'Source':<20} {'Samples':>10} {'Tokens':>15}")
    print("-" * 50)
    for src, stats in source_counts.items():
        print(f"{src:<20} {stats['samples']:>10,} {stats['tokens']:>15,}")

    return source_counts

def compare_stats(stat_list):
    """Pretty print a comparison of multiple dataset stats"""
    print("\n=== Dataset Comparison ===")
    print(f"{'Dataset':<30} {'Samples':>10} {'Tokens':>10} {'Documents':>10} {'Classes':>10}")
    print("-" * 70)
    for stat in stat_list:
        print(f"{stat['label']:<30} {stat['samples']:>10} {stat['tokens']:>10} {stat['documents']:>10} {stat['classes']:>10}")

# File paths

# Sentence pairs path
sentence_pairs_raw_file = "data/combined_sentence_pairs_300k_each.en-es"  # add _version# if needed
sentence_pairs_meta_file = "data/sentence_pairs_metadata.jsonl"  # add _version# if needed

# Travel info paths
wikivoyage_folder = "data/wikivoyage/scraped_cities_data"
travel_info_raw_file = "data/chunked_travel_info_orig_data.jsonl"  # add _version# if needed
travel_info_meta_file = "data/chunked_travel_info_metadata.jsonl"  # add _version# if needed

# Configs
sentence_pairs_config = {
    "text_fields": ["en", "es"],
    "doc_field": None,
    "class_field": "source"
}

wikivoyage_original_config = {
    "text_fields": ["en", "es"],  # in case either field exists
    "doc_field": "city",
    "class_field": "city"
}

chunked_config = {
    "text_fields": ["text"],
    "doc_field": "chunk_id",
    "class_field": "city"
}

travel_info_config = {
    "text_fields": ["text"],
    "doc_field": "chunk_id",
    "class_field": "city"
}

# Run Stats

# Analyze sentence pairs by source
source_stats = analyze_by_source(sentence_pairs_meta_file)

# Analyze sentence pairs metadata
stats_sentence_pairs = analyze_jsonl(sentence_pairs_meta_file, "Sentence Pairs Metadata", sentence_pairs_config)

# Analyze Wikivoyage original scraped city files
stats_wikivoyage_original = analyze_folder(wikivoyage_folder, "Wikivoyage Original", wikivoyage_original_config)

# Analyze Travel Info - raw chunked file
stats_travel_info_raw = analyze_jsonl(travel_info_raw_file, "Travel Info Raw Chunked", travel_info_config)

# Analyze Travel Info - metadata aligned with FAISS
stats_travel_info_meta = analyze_jsonl(travel_info_meta_file, "Travel Info Metadata", travel_info_config)

# Compare All Datasets
compare_stats([
    stats_sentence_pairs,
    stats_wikivoyage_original,
    stats_travel_info_raw,
    stats_travel_info_meta
])