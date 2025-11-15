# lazy_random_walk_rerank.py
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedding import embed_chunk

def lazy_random_walk_rerank(
    query: str,
    retrieved_chunks: List[str],
    top_k: int = 3,
    steps: int = 20,
    lazy_prob: float = 0.5,
    seed: int = 42
) -> List[str]:
    """
    使用「懶惰隨機漫步」進行 RAG 重排序（適合 HotpotQA 多跳推理）

    原理：
        1. 從「最像 query 的 chunk」開始漫步
        2. 每一步：
           - lazy_prob 機率：留在原地（強化當前重要性）
           - (1-lazy_prob) 機率：跳到相似 chunk
        3. 走過越多次的 chunk → 越重要

    參數：
        query: 使用者問題
        retrieved_chunks: 初步檢索的文本塊
        top_k: 最終返回數量
        steps: 漫步步數（建議 10~50）
        lazy_prob: 懶惰機率（0.5 為標準懶人走法）
        seed: 隨機種子（可重現）

    返回：
        重排序後的前 top_k 個文本塊
    """
    if not retrieved_chunks:
        return []

    np.random.seed(seed)

    # Step 1: 嵌入所有 chunk
    embeddings = np.array([embed_chunk(chunk) for chunk in retrieved_chunks])
    n = len(retrieved_chunks)

    # Step 2: 計算相似度轉移矩陣
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)  # 移除自環
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim_matrix / row_sums  # P(j|i) = 從 i 跳到 j

    # Step 3: 找起點（最像 query 的 chunk）
    query_emb = embed_chunk(query)
    query_sim = cosine_similarity([query_emb], embeddings)[0]
    current_idx = int(np.argmax(query_sim))

    # Step 4: 懶惰隨機漫步
    visit_count = np.zeros(n)
    for _ in range(steps):
        visit_count[current_idx] += 1

        if np.random.random() < lazy_prob:
            # 懶惰：留在原地
            continue
        else:
            # 跳到鄰居
            current_idx = np.random.choice(n, p=transition[current_idx])

    # Step 5: 依訪問次數排序
    ranked_indices = np.argsort(visit_count)[::-1]
    top_indices = ranked_indices[:top_k]
    reranked = [retrieved_chunks[i] for i in top_indices]

    return reranked