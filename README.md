# Multilingual RAG Travel Chatbot

Author: Jade Oakes
Email: jadeoakes@brandeis.edu
Date: May 12, 2025

This project implements a multilingual chatbot designed to assist with both travel-related and language learning queries in English and Spanish. The system uses Retrieval-Augmented Generation (RAG) to improve factual accuracy and contextual grounding. It retrieves relevant passages from multilingual sentence pairs and travel-specific content (scraped from Wikivoyage) to guide generation using an LLM (Mistral).

---

## Directory Structure

```
Deliverables/
├── chatbot/         # Source code for chatbot, FAISS prep, Streamlit app, experiments
├── data/            # EMPTY - See data section below for all processed data files and FAISS indexes
├── results/         # Evaluation outputs, graphs, and analysis scripts
├── Oakes_Jade_Final_Report.pdf  # Final report document
├── requirements.txt # Python dependencies
```

### chatbot/

* `experiment_queries.json`: Input prompts used during prompt experiments
* `prompt_templates.json`: Collection of prompt formats
* `prompt_loader.py`: Loads and parses prompt templates
* `multilingual_rag_travel_chatbot_app.py`: Streamlit app for interactive chatbot
* `multilingual_rag_chatbot_llm.py`: Shared LLM generation module
* `*_data_prep.py`: Prepare sentence pairs and travel passages for FAISS
* `*_faiss.py`: Build and inspect FAISS indexes
* `*_chunk_data.py`: Create fixed-length chunks for travel passages
* `length_stats_*.py`: Analyze average input and chunk lengths
* `data_stats.py`: Summary statistics of datasets
* `view_faiss_contents.py`, `inspect_faiss.py`: Debug or visualize FAISS index content
* `run_all_experiments.py`, `run_prompt_experiments.py`: Scripts to run decoding experiments

### data/

Due to its large size, this folder is empty in GitHub.

The data for this project can be accessed here:
https://drive.google.com/drive/folders/18t53ZCUy5bNl_2W6Lu6eJ4iQ-jrFe72L?usp=sharing

* Contains both subfolders (raw/processed data) and final `.jsonl` + `.faiss` files:

  * `opensubtitles/`, `tatoeba/`, `wikimatrix/`: Raw parallel corpora by source
  * `wikivoyage/scraped_cities_data/`: Travel passages scraped from Wikivoyage
  * `combined_sentence_pairs_300k_each.en-es.jsonl`: Filtered sentence pair dataset
  * `combined_travel_data.jsonl`: Merged and cleaned travel passages
  * `chunked_travel_info_orig_data.jsonl`: Unfiltered travel text before chunking
  * `chunked_travel_info_metadata.jsonl`: Metadata for each passage chunk
  * `chunked_travel_info_index.faiss`: FAISS index for chunked travel passages
  * `sentence_pairs_metadata.jsonl`: Metadata for all sentence pair entries
  * `sentence_pairs_index.faiss`: FAISS index for sentence pair embeddings

### results/

* `run_1/`: Raw model outputs in `.jsonl`, annotated evaluations, and CSV summaries
* `graphs/`: Output figures for comparisons by mode and generation strategy
* `analyze_results_modes.py`: Evaluation analysis grouped by chatbot mode
* `analyze_results_settings.py`: Evaluation analysis grouped by generation strategy
* `analyze_results_combined.py`: Combined overview of all evaluation results
* `create_results_graphs.py`: Generates bar charts from evaluation metrics
* `display_results_csv.py`: Formats annotated results into tables
* `retrieval_metrics.py`: Calculates cosine similarity and related retrieval metrics

---

## How to Run the Code (Two Modes)

### 1. Clone the repo or access the Deliverables directory

Ensure all files and folders are kept in the original structure.

### 2. Set up your environment

```bash
conda create -n rag-chatbot python=3.11
conda activate rag-chatbot
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face (for model loading)

Set your token as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

### 4. Option A: Launch the Streamlit chatbot (interactive)

This mode provides a user-friendly interface for interacting with the model live.

```bash
cd chatbot
streamlit run multilingual_rag_travel_chatbot_app.py
```

### 5. Option B: Run batch experiments (automated)

This mode executes a batch of predefined queries and saves outputs for evaluation.

**Note:** By default, all output files will be written to the existing `results/run_1/` folder and may overwrite previous outputs. If you want to preserve past runs, manually create a new folder (e.g., `run_2`) and update the save paths in the experiment scripts before execution.

```bash
python run_all_experiments.py
# or for specific prompt comparison:
python run_prompt_experiments.py
```

---

## Evaluation & Results

* Human evaluation results and annotated outputs are in `results/run_1/`
* Visual comparisons (by mode and decoding strategy) are in `results/graphs/`
* Use `create_results_graphs.py` or `analyze_results_*.py` to reproduce analysis

---

## Notes

* Do **not** commit your Hugging Face token. Use environment variables or a `.env` file (with `python-dotenv` if needed).
* Large `.jsonl` and `.faiss` files are already included in `data/`, so no extra downloads are required.

---

## Citation

This project was developed as part of the final deliverables for COSI 232B: Advanced Machine Learning Methods for NLP at Brandeis University, Spring 2025.