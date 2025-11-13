# rag/evaluate_rag.py
import json
from Ai import generate
from vectordatabase import retrieve
from encoder import rerank
from evaluation import QAExactMatch, QAF1Score 

with open("eval_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
gold_answers = [item["gold_answers"] for item in data]

em_metric = QAExactMatch()
f1_metric = QAF1Score()

preds = []

print("\n=== 开始推理 ===")
for q in questions:
    # 1. 检索
    retrieved = retrieve(q, top_k=5)
    # 2. 重排序
    reranked = rerank(q, retrieved, top_k=3)
    # 3. 生成答案
    ans = generate(q, reranked)
    preds.append(ans)
    print(f"Q: {q}\nA: {ans}\n")

em_res, _ = em_metric.calculate_metric_scores(gold_answers, preds)
f1_res, _ = f1_metric.calculate_metric_scores(gold_answers, preds)

print("\n=== 评估结果 ===")
print(f"Exact Match : {em_res['ExactMatch']:.3f}")
print(f"F1 Score    : {f1_res['F1']:.3f}")