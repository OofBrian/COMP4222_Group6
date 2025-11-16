from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .embedding import embed_chunk

def pagerank_rerank(query: str, retrieved_chunks: List[str], top_k: int = 3, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> List[str]:
    if not retrieved_chunks:
        return []

    texts = [query] + retrieved_chunks
    embeddings = np.array([embed_chunk(text) for text in texts])

    sim_matrix = cosine_similarity(embeddings)
    
    np.fill_diagonal(sim_matrix, 0)

    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = sim_matrix / row_sums  

    n = len(texts)
    pr = np.ones(n) / n

    for _ in range(max_iter):
        pr_new = (1 - alpha) / n + alpha * transition_matrix.T @ pr
        if np.linalg.norm(pr_new - pr, 1) < tol:
            pr = pr_new
            break
        pr = pr_new

    chunk_scores = pr[1:]
    ranked_indices = np.argsort(chunk_scores)[::-1]

    top_indices = ranked_indices[:top_k]
    reranked_chunks = [retrieved_chunks[i] for i in top_indices]

    return reranked_chunks