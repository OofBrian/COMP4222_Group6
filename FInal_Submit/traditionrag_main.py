import os
import json
import chromadb
from typing import List, Tuple, Dict, Any
from Traditional_RAG.Ai import generate  
from Traditional_RAG.embedding import embed_chunk
from Traditional_RAG.rerank import rerank
from Traditional_RAG.pagerank import pagerank_rerank
from chromadb.errors import NotFoundError
from eval_functions import evaluate_qa

# === Collection Setup ===
chromadb_client = chromadb.PersistentClient("./chroma.db")
TEMP_COLLECTION_NAME = "hotpotqa_temp_eval"
def get_temp_collection():
    try:
        chromadb_client.delete_collection(name=TEMP_COLLECTION_NAME)
    except NotFoundError:
        pass 

    return chromadb_client.create_collection(name=TEMP_COLLECTION_NAME)

# === Index Context ===
def index_context(context: List[Tuple[str, List[str]]], collection) -> None:
    chunks = []
    embeddings = []
    ids = []
    metadatas = []

    for title_idx, (title, sentences) in enumerate(context):
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            chunk_id = f"{title_idx}_{sent_idx}"
            chunks.append(sentence)
            embeddings.append(embed_chunk(sentence))
            ids.append(chunk_id)
            metadatas.append({"title": title, "sent_idx": sent_idx, "title_idx": title_idx})

    if chunks:
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )


# === Retrieve ===
def retrieve(query: str, collection, top_k: int = 10) -> List[str]:
    q_emb = embed_chunk(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = results['documents'][0]
    return docs


# === Main Evaluation ===
def main():
    with open("Traditional_RAG/dataset/30_hotpotqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    answer_preds = []
    answer_golds = []
    sf_metrics = {"exactMatch": [], "f1": []}
    print("\n=== HotpotQA RAG Evaluation ===")
    collection = get_temp_collection()

    for idx, item in enumerate(data):
        q = item["question"]
        answer = item["answer"]
        context = item["context"] 
        supporting_facts = item.get("supporting_facts", []) 

        answer_golds.append([answer])
        
        # --- Index context ---
        index_context(context, collection)

        # --- Retrieve ---
        retrieved_chunks = retrieve(q, collection, top_k=10)

        # --- Rerank ---
        reranked_chunks = pagerank_rerank(q, retrieved_chunks, top_k=3)

        # --- Generate ---
        generated_answer = generate(q, reranked_chunks)
        answer_preds.append(generated_answer)

        # --- Supporting Facts Evaluation ---
        print(f"\n[{idx+1}/{len(data)}] Q: {q}")
        print(f"   Gold: {answer}")
        print(f"   Pred: {generated_answer}")

        # Reset for next question
        collection = get_temp_collection()

    # === Final Metrics ===
    sf_eval, _ = evaluate_qa(answer_golds, answer_preds)
    ExactMatch=sf_eval["ExactMatch"]
    F1=sf_eval["F1"]
    print(f"ExactMatch: {ExactMatch}")
    print(f"F1: {F1}")


if __name__ == "__main__":
    main()