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
| **Dataset** | HotpotQA (30-sample subset for evaluation) |
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
