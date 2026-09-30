"""
evals/eval_correctness.py
=========================
End-to-end CORRECTNESS eval of the RAG chatbot using a hand-rolled
LLM-as-judge prompt (0-10 score) instead of a DeepEval built-in metric.

The judge compares the chatbot's ACTUAL ANSWER against the EXPECTED ANSWER
from the golden dataset and decides how factually correct it is.

    python -m evals.eval_correctness

VARIANCE
--------
A single judge call is a sample from a probability distribution, not a fixed
number: the same (question, expected, actual) can come back as 7 one time and
8 the next, because the model's top two candidate tokens are nearly tied
("probability jitter"). To make that visible we call the judge N_REPEATS times
per test case and report mean / std / range per case. A high std on a case
means that score is not trustworthy on its own.
"""

import re
import statistics
import sys

from dotenv import load_dotenv

# Windows consoles default to cp1252
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from src.ollama_llm import OllamaCloudLLM
from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"   # has query + ideal_answer
JUDGE_MODEL = OllamaCloudLLM()
N_REPEATS = 3          # judge calls per test case, to measure variance
PASS_THRESHOLD = 7     # score (0-10) at or above this counts as correct

CORRECTNESS_PROMPT = """You are evaluating whether an AI's answer is correct.

You will be given:
- QUESTION: the question that was asked
- EXPECTED ANSWER: the correct reference answer
- ACTUAL ANSWER: the answer the AI actually gave

Compare the ACTUAL ANSWER against the EXPECTED ANSWER and decide how factually correct it is.

Give a score from 0 to 10, where:
- 10 = fully correct
- 0 = completely wrong

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
ACTUAL ANSWER: {actual_answer}

Score:"""


def parse_score(text: str) -> int:
    """Pull the 0-10 integer out of the judge's reply (handles '8', 'Score: 8', '8/10')."""
    match = re.search(r"\b(10|[0-9])\b", text)
    if not match:
        raise ValueError(f"Could not find a 0-10 score in judge output: {text!r}")
    return int(match.group(1))


def judge_correctness(question: str, expected_answer: str, actual_answer: str) -> int:
    """One judge call -> one 0-10 score."""
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )
    return parse_score(JUDGE_MODEL.generate(prompt))


def run(rag, n_repeats: int = N_REPEATS):
    goldens = load_goldens(GOLDEN_PATH)

    rows = []
    for g in goldens:
        actual = rag.invoke(g["query"])["answer"]          # retrieve -> rerank -> generate

        scores = [
            judge_correctness(g["query"], g["ideal_answer"], actual)
            for _ in range(n_repeats)
        ]
        rows.append({
            "id": g["id"],
            "query": g["query"],
            "scores": scores,
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores),
            "spread": max(scores) - min(scores),
        })
        print(f"{g['id']}: scores={scores}  mean={rows[-1]['mean']:.1f}  std={rows[-1]['std']:.2f}")

    return rows


def summarize(rows):
    """Same shape as harness.summarize_by_metric, plus judge-variance stats."""
    means = [r["mean"] for r in rows]
    passed = sum(1 for m in means if m >= PASS_THRESHOLD)
    return {
        "correctness": {
            "n": len(rows),
            "pass_rate": 100 * passed / len(rows) if rows else 0.0,
            "avg_score": statistics.mean(means) / 10 if means else float("nan"),   # 0-1 like DeepEval
            "min_score": min(means) / 10 if means else float("nan"),
            "max_score": max(means) / 10 if means else float("nan"),
            "avg_judge_std": statistics.mean(r["std"] for r in rows) if rows else float("nan"),
            "unstable_cases": sum(1 for r in rows if r["spread"] >= 2),
        }
    }


def print_report(summary):
    s = summary["correctness"]
    print("\n" + "=" * 60)
    print("correctness  (LLM-as-judge, 0-10)")
    print("=" * 60)
    print(f"  cases            : {s['n']}")
    print(f"  pass rate (>={PASS_THRESHOLD}) : {s['pass_rate']:.0f}%")
    print(f"  avg score        : {s['avg_score'] * 10:.2f} / 10")
    print(f"  avg judge std    : {s['avg_judge_std']:.2f}   (judge variance across {N_REPEATS} repeats)")
    print(f"  unstable cases   : {s['unstable_cases']}   (repeat scores differ by >= 2 points)")
    print("=" * 60)


if __name__ == "__main__":
    rows = run(RagPipeline())
    print_report(summarize(rows))
