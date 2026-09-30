from dotenv import load_dotenv

from deepeval import evaluate
from src.ollama_llm import OllamaCloudLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from src.reranker import RerankingRetriever
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)


# ============================================================
# ENV
# ============================================================

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"

THRESHOLD = 0.7

OLLAMA_MODEL = "gpt-oss:20b"


# ============================================================
# OLLAMA CLOUD LLM
# ============================================================

# Shared free-model wrapper (Ollama cloud, no OpenAI key needed) — see src/ollama_llm.py


# ============================================================
# JUDGE
# ============================================================

JUDGE_MODEL = OllamaCloudLLM()


# ============================================================
# EVALUATION
# ============================================================

def run(retriever):

    # 1. Load fixed golden dataset
    goldens = load_goldens(GOLDEN_PATH)

    print(f"\nLoaded {len(goldens)} golden test cases.")

    # 2. Run retriever
    test_cases = []

    for i, g in enumerate(goldens, 1):

        print(
            f"Running retriever "
            f"{i}/{len(goldens)}..."
        )

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
                actual_output="(generator not evaluated)",
            )
        )

    # 3. Metrics
    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # 4. Evaluate
    print("\nStarting DeepEval...")
    print(
        f"Judge: {JUDGE_MODEL.get_model_name()}"
    )

    result = evaluate(
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

            "judge_model": OLLAMA_MODEL,

            "judge_provider": "ollama-cloud",

            "golden_set": GOLDEN_PATH,
        },
    )

    return summarize_by_metric(result)


# ============================================================
# LOCAL RUN
# ============================================================

def run_local():

    retriever = RerankingRetriever(
        fetch_k=10,
        top_k=5,
    )

    return run(retriever)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RAG RETRIEVER EVALUATION")
    print("=" * 60)

    print(f"Judge model : {OLLAMA_MODEL}")
    print(f"Golden set  : {GOLDEN_PATH}")
    print(f"Threshold   : {THRESHOLD}")

    summary = run_local()

    print_summary(
        "retriever",
        summary,
    )