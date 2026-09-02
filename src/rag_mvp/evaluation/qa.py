from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

import httpx

from rag_mvp.qa.grounding import lexical_support


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percent))
    return ordered[index]


def faithfulness(answer: str, source_texts: list[str]) -> float:
    """Fraction of answer content tokens supported by at least one retrieved passage."""
    return lexical_support(answer, source_texts)


def evaluate_case(case: dict[str, object], response: dict[str, object]) -> dict[str, object]:
    expected_refusal = bool(case.get("should_refuse", False))
    refused = bool(response["refused"])
    answer = str(response["answer"])
    terms = [str(term).casefold() for term in case.get("expected_terms", [])]
    compliant = refused == expected_refusal and (
        expected_refusal or (all(term in answer.casefold() for term in terms) and "[" in answer)
    )
    same_language = bool(re.search(r"[\u4e00-\u9fff]", answer)) == bool(
        re.search(r"[\u4e00-\u9fff]", str(case["query"]))
    )
    style = expected_refusal or (
        same_language and bool(re.search(r"\[\d+\]", answer)) and "<think>" not in answer.casefold()
    )
    source_texts = [str(item.get("text", "")) for item in response.get("sources", [])]
    return {
        "id": case["id"],
        "expected_refusal": expected_refusal,
        "compliant": compliant,
        "style_consistent": style,
        "refusal_correct": refused == expected_refusal,
        "faithfulness": 1.0 if expected_refusal else faithfulness(answer, source_texts),
        "total_ms": float(response["total_ms"]),
        "tokens": int(response["token_usage"]["total_tokens"]),
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the running grounded QA API")
    parser.add_argument("--dataset", default="evals/qa_cases.json")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/qa")
    parser.add_argument("--output", default="evals/reports/qa_baseline")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    details = []
    with httpx.Client(timeout=180) as client:
        for case in cases:
            response = client.post(args.endpoint, json={"query": case["query"], "top_k": 3})
            response.raise_for_status()
            details.append(evaluate_case(case, response.json()))

    count = len(details)
    answerable = [item for item in details if not item["expected_refusal"]]
    latencies = [float(item["total_ms"]) for item in details]
    summary = {
        "cases": count,
        "faithfulness": sum(float(item["faithfulness"]) for item in answerable)
        / max(1, len(answerable)),
        "answer_compliance": sum(bool(item["compliant"]) for item in details) / count,
        "style_consistency": sum(bool(item["style_consistent"]) for item in details) / count,
        "refusal_appropriateness": sum(bool(item["refusal_correct"]) for item in details) / count,
        "p50_ms": median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "requests_within_10s": sum(value <= 10_000 for value in latencies) / count,
        "total_tokens": sum(int(item["tokens"]) for item in details),
        "estimated_api_cost_per_1000_calls_usd": 0.0,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = "# QA evaluation\n\n" + "\n".join(
        f"- {key.replace('_', ' ').title()}: `{value:.3f}`" if isinstance(value, float) else f"- {key}: `{value}`"
        for key, value in summary.items()
    )
    (output / "report.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
