import os

from openai import AsyncOpenAI, OpenAI
from deepeval.models import DeepEvalBaseLLM


class OllamaCloudLLM(DeepEvalBaseLLM):
    """
    Free LLM wrapper for DeepEval, backed by Ollama's cloud API (OpenAI-
    compatible endpoint, gpt-oss:20b by default) instead of a paid OpenAI key.

    Used as:
      - the LLM judge for eval metrics (ContextualRecallMetric, etc.)
      - the generator model for DeepEval's Synthesizer, so golden-dataset
        generation doesn't need OPENAI_API_KEY either.

    generate()/a_generate() deliberately do NOT accept a `schema` kwarg.
    DeepEval always tries model.generate(prompt, schema=schema) first and
    catches the TypeError that raises to fall back to plain-text generation
    + its own JSON-extraction/retry logic (trimAndLoadJson). That fallback
    is more robust than hand-rolling structured output ourselves, since this
    model has no native JSON-mode/schema API the way OpenAI's does. If we
    accepted **schema and silently ignored it instead, DeepEval would think
    we returned a validated schema object when we actually returned a plain
    string, and break downstream (this bit the Synthesizer specifically,
    which is stricter about this than the metrics path).
    """

    def __init__(self, model="gpt-oss:20b"):
        self.model = model
        self.client = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=os.environ["OLLAMA_API_KEY"],
        )
        self.async_client = AsyncOpenAI(
            base_url="https://ollama.com/v1",
            api_key=os.environ["OLLAMA_API_KEY"],
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Ollama returned an empty response.")
        return content

    async def a_generate(self, prompt: str) -> str:
        # Real async client so DeepEval's async paths (metrics' a_measure,
        # Synthesizer's async_mode=True) run judge calls concurrently instead
        # of one-at-a-time.
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Ollama returned an empty response.")
        return content

    def get_model_name(self):
        return f"Ollama Cloud - {self.model}"
