# COMP4222_Group6
# Evaluation Scripts

This repository contains scripts for evaluating models with and without RAG (Retrieval-Augmented Generation). The following scripts are provided:

## Main Files

1. **`No_Rag_main.py`**
   - This script is used for evaluation **without RAG**.

2. **`Hipporag_main.py`**
   - This script is used for evaluation **with HippoRAG**.

## Instructions

1. Navigate to the directory containing the scripts.

2. Set the environment variables:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export OPENAI_API_KEY=<your_openai_api_key>  # Replace <your_openai_api_key> with your DeepSeek API key

3. Running the scripts:
   
   - For No_Rag_main.py:
   Open the file and manually update the dataset name inside the script to the desired dataset.
   Run the script:
   ```bash
   python No_Rag_main.py
   ```
   
   - For Hipporag_main.py:
   Run the script directly by specifying the dataset name as a command-line argument:
   ```bash
   python Hipporag_main.py --<dataset_name>
   ```
4. Running experiments:

   - To run experiments with custom configurations, modify the configs in /src/hipporag/utils
   
# Traditional RAG System with PageRank Reranking

A lightweight, modular **Retrieval-Augmented Generation (RAG)** pipeline for **multi-hop QA** using the **HotpotQA** dataset. This system combines:

- **Dense retrieval** with `text2vec-base-chinese` embeddings
- **Graph-based reranking** using **PageRank on query-chunk similarity**
- **Cross-encoder reranking** (alternative option)
- **Generation** via **DeepSeek API**
- **Evaluation** with Exact Match (EM) and F1 on answer spans

Built for **Chinese/English bilingual support**, local persistence, and easy experimentation.

---

## Features

| Feature | Description |
|-------|-----------|
| **Embedding Model** | `shibing624/text2vec-base-chinese` (bilingual) |
| **Vector DB** | ChromaDB (persistent, lightweight) |
| **Reranking** | PageRank on cosine similarity graph + Cross-Encoder option |
| **LLM** | DeepSeek (`deepseek-chat`) via API |
| **Dataset** | HotpotQA |
| **Evaluation** | Exact Match & F1 (answer-level) |

---


### Final Checklist

```bash
# 1. Setup
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt

# 2. API Key
$env: DEEPSEEK_API_KEY = "your_openai_api_key"

# 3. Run
python traditionrag_main.py
```

**Remark**:
  > **If you want to use your own dataset?**  
  > Replace the path:
  > ```python:disable-run
  > with open("your own path", "r", encoding="utf-8") as f:
  > ```
  > with your own path and ensure JSON format matches.

## Note: 
All available datasets for evaluation are stored in the /reproduce/dataset directory. You only need to include the dataset name in the script or command.
