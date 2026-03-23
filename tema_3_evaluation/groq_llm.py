import os

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = os.environ.get("GROQ_MODEL", model)
        self.max_tokens = int(os.environ.get("EVAL_LLM_MAX_TOKENS", "1024"))

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Seteaza GROQ_API_KEY in .env pentru evaluare.")

        # Foloseste endpoint-ul configurat in .env (Groq/OpenRouter/API compatibil OpenAI).
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model
