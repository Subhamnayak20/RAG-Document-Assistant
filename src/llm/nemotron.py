
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError("NVIDIA_API_KEY is missing from your environment.")

client = OpenAI(
    api_key=API_KEY,
    base_url=os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1"
    )
)

MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)


def generate_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Use only the supplied context. "
                    "If the answer is unavailable, say so. "
                    "Never invent information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Give a clear and concise answer."
                )
            }
        ],
        temperature=0.2,
        max_tokens=2048,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    if not response.choices:
        raise RuntimeError("Nemotron returned no response choices.")

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError(
            "Nemotron returned an empty answer. Please retry."
        )

    return answer.strip()
