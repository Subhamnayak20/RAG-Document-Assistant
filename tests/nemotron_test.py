import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL"),
    api_key=os.getenv("NVIDIA_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("NVIDIA_MODEL"),
    messages=[
        {
            "role": "user",
            "content": "Explain Retrieval Augmented Generation in simple terms."
        }
    ],
    temperature=0.6,
    max_tokens=500,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

print(response.choices[0].message.content)