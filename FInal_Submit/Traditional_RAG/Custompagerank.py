# Traditional_RAG/pagerank.py
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import math
import random

from Traditional_RAG.embedding import embed_chunk


class VariablePPRConfig:
    """Configuration mirroring the 4222 project fields"""
    def __init__(
        self,
        use_variable_alpha: bool = True,
        variable_alpha_base: float = 0.40,      # α for seeds (distance = 0)
        variable_alpha_low: float = 0.10,       # α for 1-hop nodes
        variable_alpha_high: float = 0.10,      # α for far nodes (≥ k_hop)
        variable_alpha_k_hop: int = 3,          # threshold for switching to high α
        variable_ppr_num_walks: int = 5000,     # MC walks (accuracy vs speed)
        variable_ppr_max_hops: int = 20,        # safety cap
    ):
        self.use_variable_alpha = use_variable_alpha
        self.variable_alpha_base = variable_alpha_base
        self.variable_alpha_low = variable_alpha_low
        self.variable_alpha_high = variable_alpha_high
        self.variable_alpha_k_hop = variable_alpha_k_hop
        self.variable_ppr_num_walks = variable_ppr_num_walks
        self.variable_ppr_max_hops = variable_ppr_max_hops


# Default config (you can tune this)
PPR_CONFIG = VariablePPRConfig()


def weightpagerank_rerank(
    query: str,
    retrieved_chunks: List[str],
    top_k: int = 3,
    config: VariablePPRConfig = PPR_CONFIG
) -> List[str]:
    """
    Rerank retrieved chunks using custom distance-dependent Personalized PageRank.
    Seeds = query + retrieved chunks (query has highest reset probability).
    """
    if not retrieved_chunks:
        return []

    # 1. Texts: query is the main seed, chunks are nodes
    texts = [query] + retrieved_chunks
    n = len(texts)

    # 2. Build similarity graph (cosine similarity → edge weights)
    embeddings = np.array([embed_chunk(text) for text in texts])
    sim_matrix = cosine_similarity(embeddings)  # (n, n)

    # Remove self-loops
    np.fill_diagonal(sim_matrix, 0)

    # Convert to adjacency list with weights
    neighbors = [[] for _ in range(n)]
    weights = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if sim_matrix[i, j] > 0:
                neighbors[i].append(j)
                weights[i].append(sim_matrix[i, j])

    # 3. Seed (reset) probabilities: query gets 90%, chunks share 10%
    reset_prob = np.zeros(n)
    reset_prob[0] = 0.9                    # query is primary seed
    reset_prob[1:] = 0.1 / len(retrieved_chunks)  # uniform over chunks

    if not config.use_variable_alpha:
        # Fallback to classic power-iteration PPR (your old version)
        return _classic_pagerank_rerank(query, retrieved_chunks, top_k)

    # ====================== CUSTOM VARIABLE-α PPR (4222 style) ======================
    # 3. BFS from seeds to compute min distance to any seed
    seeds = [i for i in range(n) if reset_prob[i] > 0]
    dist = [math.inf] * n
    for s in seeds:
        dist[s] = 0
    q = deque(seeds)
    visited = set(seeds)
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if v not in visited:
                visited.add(v)
                dist[v] = dist[u] + 1
                q.append(v)

    # 4. Distance-dependent teleport probability α(d)
    def alpha_d(d: float) -> float:
        if d == 0:  # seed
            return config.variable_alpha_base
        if d == 1:
            return config.variable_alpha_low
        if d >= config.variable_alpha_k_hop:
            return config.variable_alpha_high
        # Linear interpolation between low and high for 1 < d < k_hop
        frac = (d - 1) / (config.variable_alpha_k_hop - 1)
        return config.variable_alpha_low + frac * (config.variable_alpha_high - config.variable_alpha_low)

    # 5. Monte-Carlo random walks
    scores = np.zeros(n)
    num_walks = config.variable_ppr_num_walks
    max_hops = config.variable_ppr_max_hops

    for _ in range(num_walks):
        # Start from seed distribution
        cur = random.choices(range(n), weights=reset_prob, k=1)[0]

        for _ in range(max_hops):
            d = dist[cur]
            if random.random() < alpha_d(d):
                scores[cur] += 1.0 / num_walks
                break

            if not neighbors[cur]:  # dangling node
                scores[cur] += 1.0 / num_walks
                break

            # Weighted random walk
            total_w = sum(weights[cur])
            if total_w <= 0:
                break
            probs = [w / total_w for w in weights[cur]]
            cur = random.choices(neighbors[cur], weights=probs, k=1)[0]
        else:
            # Reached max_hops → absorb at current node
            scores[cur] += 1.0 / num_walks

    # 6. Return top-k chunks (exclude query itself, i.e. index 0)
    chunk_scores = scores[1:]                                    
    ranked_indices = np.argsort(chunk_scores)[::-1][:top_k]     
    reranked_chunks = [retrieved_chunks[i] for i in ranked_indices]
    return reranked_chunks


# Optional: keep old classic implementation as fallback
def _classic_pagerank_rerank(query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
    texts = [query] + retrieved_chunks
    embeddings = np.array([embed_chunk(text) for text in texts])
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim_matrix / row_sums

    n = len(texts)
    pr = np.ones(n) / n
    alpha = 0.85
    for _ in range(100):
        pr_new = (1 - alpha) / n + alpha * transition.T @ pr
        if np.linalg.norm(pr_new - pr, 1) < 1e-6:
            break
        pr = pr_new

    chunk_scores = pr[1:]
    ranked = np.argsort(chunk_scores)[::-1][:top_k]
    return [retrieved_chunks[i] for i in ranked]