# rag/evaluate_rag.py
import json
import os
import chromadb
from typing import List, Tuple, Dict, Any
from Ai import generate
from embedding import embed_chunk
from encoder import rerank
from evaluation import QAExactMatch, QAF1Score
from chromadb.errors import NotFoundError

from eval import evaluate_qa

# === HotpotQA Supporting Facts Evaluation ===
def supporting_facts_precision_recall_f1(pred_chunks: List[str], gold_sfs: List[Tuple[str, int]], context: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    """
    Evaluate supporting facts based on retrieved chunks.
    We consider a sentence retrieved if it's in pred_chunks.
    """
    title_to_sentences = {title: sentences for title, sentences in context}
    
    pred_sfs = set()
    for chunk in pred_chunks:
        for title, sentences in title_to_sentences.items():
            if chunk in sentences:
                sent_idx = sentences.index(chunk)
                pred_sfs.add((title, sent_idx))
                break

    gold_sfs_set = set((title, sent_idx) for title, sent_idx in gold_sfs)

    if not gold_sfs_set:
        return {"sf_precision": 1.0, "sf_recall": 1.0, "sf_f1": 1.0}

    tp = len(pred_sfs & gold_sfs_set)
    precision = tp / len(pred_sfs) if pred_sfs else 0.0
    recall = tp / len(gold_sfs_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"sf_precision": precision, "sf_recall": recall, "sf_f1": f1}


# === Collection Setup ===
chromadb_client = chromadb.PersistentClient("./chroma.db")
TEMP_COLLECTION_NAME = "hotpotqa_temp_eval"
def get_temp_collection():
    try:
        chromadb_client.delete_collection(name=TEMP_COLLECTION_NAME)
    except NotFoundError:
        pass  # Collection didn't exist → nothing to delete

    return chromadb_client.create_collection(name=TEMP_COLLECTION_NAME)
'''
def get_temp_collection():
    chromadb_client = chromadb.PersistentClient("./chroma.db")
    return chromadb_client.get_or_create_collection(name="default")
'''

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
    with open("hotpotqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    em_metric = QAExactMatch()
    f1_metric = QAF1Score()

    answer_preds = []
    answer_golds = []
    #sf_metrics = {"precision": [], "recall": [], "f1": []}
    sf_metrics = {"exactMatch": [], "f1": []}
    print("\n=== HotpotQA RAG Evaluation ===")
    collection = get_temp_collection()

    for idx, item in enumerate(data):
        q = item["question"]
        answer = item["answer"]
        context = item["context"]  # List[title, List[sentences]]
        supporting_facts = item.get("supporting_facts", [])  # List[(title, sent_idx)]

        answer_golds.append([answer])
        
        # --- Index context ---
        index_context(context, collection)

        # --- Retrieve ---
        retrieved_chunks = retrieve(q, collection, top_k=10)

        # --- Rerank ---
        reranked_chunks = rerank(q, retrieved_chunks, top_k=3)

        # --- Generate ---
        generated_answer = generate(q, reranked_chunks)
        answer_preds.append(generated_answer)

        # --- Supporting Facts Evaluation ---
        '''
        sf_scores = supporting_facts_precision_recall_f1(reranked_chunks, supporting_facts, context)
        sf_metrics["precision"].append(sf_scores["sf_precision"])
        sf_metrics["recall"].append(sf_scores["sf_recall"])
        sf_metrics["f1"].append(sf_scores["sf_f1"])
        
        print(f"\n[{idx+1}/{len(data)}] Q: {q}")
        print(f"   Gold: {answer}")
        print(f"   Pred: {generated_answer}")
        print(f"   SF F1: {sf_scores['sf_f1']:.3f} (P:{sf_scores['sf_precision']:.3f}, R:{sf_scores['sf_recall']:.3f})")
        '''
        if(idx==30): break
        print(f"\n[{idx+1}/{len(data)}] Q: {q}")
        '''
        print(f"\n[{idx+1}/{len(data)}] Q: {q}")
        sf_eval, _ = evaluate_qa(answer_golds, answer_preds)
        test=sf_eval["ExactMatch"]
        test1=sf_eval["F1"]
        print(f"ExactMatch: {test}")
        print(f"F1: {test1}")
        '''
        

        # Reset for next question
        collection = get_temp_collection()

    # === Final Metrics ===
    '''
    em_res, _ = em_metric.calculate_metric_scores(answer_golds, answer_preds)
    f1_res, _ = f1_metric.calculate_metric_scores(answer_golds, answer_preds)

    avg_sf_p = sum(sf_metrics["precision"]) / len(sf_metrics["precision"])
    avg_sf_r = sum(sf_metrics["recall"]) / len(sf_metrics["recall"])
    avg_sf_f1 = sum(sf_metrics["f1"]) / len(sf_metrics["f1"])

    print("\n" + "="*50)
    print(" HOTPOTQA EVALUATION RESULTS ")
    print("="*50)
    print(f"Answer Exact Match : {em_res['ExactMatch']:.3f}")
    print(f"Answer F1 Score    : {f1_res['F1']:.3f}")
    print(f"Supp. Facts Prec.  : {avg_sf_p:.3f}")
    print(f"Supp. Facts Recall : {avg_sf_r:.3f}")
    print(f"Supp. Facts F1     : {avg_sf_f1:.3f}")
    print(f"Joint F1 (Ans + SF): {f1_res['F1'] * avg_sf_f1:.3f}")
    print("="*50)
    '''
    sf_eval, _ = evaluate_qa(answer_golds, answer_preds)
    ExactMatch=sf_eval["ExactMatch"]
    F1=sf_eval["F1"]
    print(f"ExactMatch: {ExactMatch}")
    print(f"F1: {F1}")


if __name__ == "__main__":
    main()