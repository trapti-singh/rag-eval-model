import json

from dotenv import load_dotenv

from deepeval import evaluate
from src.ollama_llm import OllamaCloudLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from src.reranker import RerankingRetriever


load_dotenv()


GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "gpt-oss:20b"
THRESHOLD = 0.7


# ============================================================
# Ollama Cloud Judge
# ============================================================

# Shared free-model wrapper (Ollama cloud, no OpenAI key needed) — see src/ollama_llm.py


# Create the judge
judge = OllamaCloudLLM()


# ============================================================
# 1. LOAD GOLDEN SET
# ============================================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)


# ============================================================
# 2. RUN RERANKING RETRIEVER
# ============================================================

retriever = RerankingRetriever()

test_cases = []

for g in goldens:

    retrieved = retriever.invoke(g["query"])

    retrieval_context = [
        doc.page_content
        for doc in retrieved
    ]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# ============================================================
# 3. METRICS
# ============================================================

metrics = [

    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),

    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]


# ============================================================
# 4. EVALUATE
# ============================================================

evaluate(
    test_cases=test_cases,
    metrics=metrics,

    hyperparameters={

        "retriever": "cross_encoder_reranker",

        "embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "reranker_model":
            "cross-encoder/ms-marco-MiniLM-L-6-v2",

        "chunk_size": 1000,

        "chunk_overlap": 150,

        "fetch_k": 10,

        "top_k": 5,

        "judge_model": JUDGE_MODEL,

        "judge_provider": "ollama-cloud",

        "golden_set": GOLDEN_PATH,
    },
)