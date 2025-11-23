import os
import json
from tqdm import tqdm  # 用于进度条显示
from typing import List, Dict, Tuple, Any
from openai import OpenAI
import numpy as np
from eval_functions import evaluate_qa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# extract answer
def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def model_call(query):
    system_prompt = (
        "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
        "Your response starts after 'Thought: ', where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. "
        "Conclude with 'Answer: ' to present a concise, definitive response, devoid of additional elaborations."
    )
        
    client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ],
    stream=False
    )
    return response.choices[0].message.content

# 处理问答
def process(queries: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    对输入的查询列表运行问答流程并处理模型响应。
    """
    # 准备问答提示
    all_qa_prompts = []
    for query in tqdm(queries, desc="Collecting QA prompts"):
        prompt_user = f"Question: {query}\nThought:"
        all_qa_prompts.append(prompt_user)

    # 执行模型推理
    all_qa_results = []
    for qa_prompt in tqdm(all_qa_prompts, desc="QA Reading"):
        result = model_call(qa_prompt)
        all_qa_results.append(result)

    # 处理响应并提取答案
    queries_solutions = []
    for query, response_content in zip(queries, all_qa_results):
        try:
            # 从响应中提取预测答案
            pred_ans = response_content.split('Answer:')[1].strip()
        except Exception as e:
            print(f"Error in parsing the answer from the response: {str(e)}")
            pred_ans = response_content

        # 构建结果
        queries_solutions.append({
            "question": query,
            "predicted_answer": pred_ans
        })

    return queries_solutions, all_qa_results

def main():
    # Extract QA
    dataset_name = 'musique'
    samples = json.load(open(f"../reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]
    gold_answers = get_gold_answers(samples)

    #  # Testing case
    #  # Separate Retrieval & QA
    # all_queries = [
    #     "What is George Rankin's occupation?",
    #     "How did Cinderella reach her happy ending?",
    #     "What county is Erik Hort's birthplace a part of?"
    # ]

    # For Evaluation
    """
    gold_answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]
    """

   # run model
    print("Running QA inference...")
    queries_solutions, all_qa_results = process(all_queries)

    # 提取模型答案
    predicted_answers = [query_solution['predicted_answer'] for query_solution in queries_solutions]

    # 计算 EM 和 F1 分数
    overall_results, example_results = evaluate_qa(
    gold_answers=gold_answers,
    predicted_answers=predicted_answers
    )

    # 打印评测结果
    print("Evaluation Results:")
    print("Overall Exact Match (EM):", overall_results["ExactMatch"])
    print("Overall F1 Score:", overall_results["F1"])


if __name__ == "__main__":
    main()
