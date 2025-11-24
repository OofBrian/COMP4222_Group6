from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Set DEEPSEEK_API_KEY in your .env file")

deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/"
)


def generate(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict answer-only assistant. "
                "You MUST respond with ONLY the final answer to the question. "
                "You answers should based SOLELY on the provided context, without adding any additional information."
                "Do NOT include any explanation, reasoning, or context summary"
                "Do NOT say 'Based on the context' or 'The answer is'. "
                "If the answer is not in the context, say 'I don't know' and nothing else."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        }
    ]
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Error: {str(e)}"


