from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedding import embed_chunk

def pagerank_rerank(query: str, retrieved_chunks: List[str], top_k: int = 3, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> List[str]:
    """
    使用 PageRank 演算法對檢索出的文本塊進行重排序。

    原理：
    1. 將 query + 每個 chunk 視為圖中的節點（共 N+1 個節點）
    2. 計算節點間的餘弦相似度作為邊權重
    3. 構建轉移矩陣並執行 PageRank 迭代
    4. 根據 chunk 節點的 PR 值排序

    參數：
        query: 使用者查詢
        retrieved_chunks: 初步檢索出的文本塊列表
        top_k: 最終返回的塊數量
        alpha: PageRank 阻尼係數（預設 0.85)
        max_iter: 最大迭代次數
        tol: 收斂閾值

    返回：
        按 PageRank 分數排序後的前 top_k 個文本塊
    """
    if not retrieved_chunks:
        return []

    # Step 1: 嵌入所有文本（包含 query）
    texts = [query] + retrieved_chunks
    embeddings = np.array([embed_chunk(text) for text in texts])

    # Step 2: 計算餘弦相似度矩陣
    sim_matrix = cosine_similarity(embeddings)  # (N+1, N+1)
    
    # 避免自環影響（可選：將對角線設為 0）
    np.fill_diagonal(sim_matrix, 0)

    # Step 3: 正規化為轉移概率（列正規化）
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    # 避免除以 0
    row_sums[row_sums == 0] = 1.0
    transition_matrix = sim_matrix / row_sums  # M[i, j] = 從 i 跳到 j 的機率

    # Step 4: PageRank 迭代
    n = len(texts)
    pr = np.ones(n) / n  # 初始均勻分佈

    for _ in range(max_iter):
        pr_new = (1 - alpha) / n + alpha * transition_matrix.T @ pr
        if np.linalg.norm(pr_new - pr, 1) < tol:
            pr = pr_new
            break
        pr = pr_new

    # Step 5: 取出 chunk 部分的 PR 分數（跳過第 0 個 query）
    chunk_scores = pr[1:]
    ranked_indices = np.argsort(chunk_scores)[::-1]  # 降冪

    # Step 6: 返回前 top_k 個 chunk
    top_indices = ranked_indices[:top_k]
    reranked_chunks = [retrieved_chunks[i] for i in top_indices]

    return reranked_chunks